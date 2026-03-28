#!/usr/bin/env python3
"""
GraspGen ZMQ Server - 6-DOF Grasp Generation Service

A ZMQ-based server that wraps NVIDIA's GraspGen diffusion model for 6-DOF
grasp generation, enabling remote clients to generate grasp poses from
point clouds or meshes.
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import zmq

# Add GraspGen to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../deps/GraspGen"))

from grasp_gen.grasp_server import GraspGenSampler, load_grasp_cfg

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("graspgen_server.log"),
    ],
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_HOST = "*"
DEFAULT_PORT = 5556
DEFAULT_GPU_ID = 0
DEFAULT_RECV_TIMEOUT = 60000  # 60 seconds for grasp generation
DEFAULT_GRIPPER_CONFIG = "graspgen_robotiq_2f_140.yml"


class GraspGenZMQServer:
    """
    ZMQ-based server for GraspGen 6-DOF grasp generation.

    This server listens for JSON requests containing point cloud data
    and returns 6-DOF grasp poses with confidence scores.
    """

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        gpu_id: int = DEFAULT_GPU_ID,
        recv_timeout: int = DEFAULT_RECV_TIMEOUT,
        gripper_config: str = DEFAULT_GRIPPER_CONFIG,
    ):
        """
        Initialize the GraspGen ZMQ server.

        Args:
            host: Host to bind to (default: "*" for all interfaces)
            port: ZMQ port to listen on
            gpu_id: GPU device ID
            recv_timeout: ZMQ receive timeout in milliseconds
            gripper_config: Gripper configuration filename
        """
        # ------------------------ ZMQ Server Configuration ------------------------
        self.host = host
        self.port = port
        self.gpu_id = gpu_id
        self.recv_timeout = recv_timeout
        self.gripper_config = gripper_config

        # ZMQ context and socket
        self.context: Optional[zmq.Context] = None
        self.socket: Optional[zmq.Socket] = None

        # ------------------------ GraspGen Model Configuration ------------------------
        config_path = os.path.join("GraspGenModels", "checkpoints", self.gripper_config)
        logger.info(f"Loading gripper config from: {config_path}")
        self.cfg = load_grasp_cfg(config_path)

        self.gripper_name = self.cfg.data.gripper_name
        self.model_name = self.cfg.eval.model_name

        # GraspGenSampler instance (loaded on startup)
        self.sampler: Optional[GraspGenSampler] = None

        # Request counter
        self.request_count = 0

        logger.info(
            f"GraspGenZMQServer initialized: host={host}, port={port}, "
            f"gripper={gripper_config}, gpu={gpu_id}"
        )

    def _setup_cuda_device(self) -> None:
        """Configure CUDA device based on configuration."""
        if not torch.cuda.is_available():
            logger.error("CUDA not available. This server requires GPU to run.")
            raise RuntimeError(
                "CUDA is not available on this system. "
                "GraspGen requires GPU acceleration. "
                "Please ensure PyTorch is installed with CUDA support."
            )

        device_count = torch.cuda.device_count()
        if self.gpu_id >= device_count:
            logger.warning(
                f"Requested GPU {self.gpu_id} not available. "
                f"Available GPUs: {device_count}. Using GPU 0."
            )
            self.gpu_id = 0

        torch.cuda.set_device(self.gpu_id)
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(self.gpu_id)}")

    def load_model(self) -> None:
        """
        Load the GraspGen model into memory.

        This should be called once during initialization.
        """
        logger.info(f"Loading GraspGen model: {self.gripper_config}")
        start_time = time.time()

        try:
            self._setup_cuda_device()

            # Resolve gripper config path
            logger.info(
                f"Initializing GraspGenSampler (model={self.model_name}, gripper={self.gripper_name})"
            )
            self.sampler = GraspGenSampler(self.cfg)

            load_time = time.time() - start_time
            logger.info(f"GraspGen model loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load GraspGen model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _validate_request(self, request: Dict[str, Any]) -> None:
        """
        Validate incoming request format and parameters.

        Args:
            request: Parsed JSON request dictionary

        Raises:
            ValueError: If request is invalid
        """
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")

        # Check required fields
        if "point_cloud" not in request:
            raise ValueError("Missing required field: 'point_cloud'")

        # Validate optional fields
        if "num_grasps" in request:
            if not isinstance(request["num_grasps"], int) or request["num_grasps"] < 1:
                raise ValueError("Field 'num_grasps' must be a positive integer")

        if "topk_num_grasps" in request:
            if not isinstance(request["topk_num_grasps"], int):
                raise ValueError("Field 'topk_num_grasps' must be an integer")

        if "grasp_threshold" in request:
            if not isinstance(request["grasp_threshold"], (int, float)):
                raise ValueError("Field 'grasp_threshold' must be a number")

        if "options" in request and not isinstance(request["options"], dict):
            raise ValueError("Field 'options' must be an object")

    def _decode_numpy_array(self, array_data: Dict[str, Any]) -> np.ndarray:
        """
        Decode a serialized numpy array from dict format.

        Args:
            array_data: Dict with keys 'data' (base64), 'shape', 'dtype'

        Returns:
            Reconstructed numpy array

        Raises:
            RuntimeError: If decoding fails
        """
        try:
            byte_data = base64.b64decode(array_data["data"])
            return np.frombuffer(byte_data, dtype=array_data["dtype"]).reshape(
                array_data["shape"]
            )
        except Exception as e:
            logger.error(f"Failed to decode numpy array: {e}")
            raise RuntimeError(f"Array decoding failed: {e}") from e

    def _encode_numpy_array(self, array: np.ndarray) -> Dict[str, Any]:
        """
        Encode numpy array to base64 format.

        Args:
            array: NumPy array

        Returns:
            Dictionary with data, shape, and dtype
        """
        return {
            "data": base64.b64encode(array.tobytes()).decode("utf-8"),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    def _generate_grasps(
        self,
        point_cloud: np.ndarray,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Generate grasp poses from point cloud using GraspGen.

        Args:
            point_cloud: (N, 3) numpy float32 array of object points
            options: Generation parameters

        Returns:
            Dictionary with grasps, confidences, and metadata

        Raises:
            RuntimeError: If generation fails
        """
        options = options or {}

        # Extract parameters
        num_grasps = options.get("num_grasps", 200)
        topk_num_grasps = options.get("topk_num_grasps", -1)
        grasp_threshold = options.get("grasp_threshold", -1.0)
        min_grasps = options.get("min_grasps", 40)
        max_tries = options.get("max_tries", 6)
        remove_outliers = options.get("remove_outliers", True)

        logger.info(
            f"Generating grasps for {len(point_cloud)} points "
            f"(num_grasps={num_grasps}, topk={topk_num_grasps}, threshold={grasp_threshold})"
        )

        start_time = time.time()

        try:
            # Run GraspGen inference
            grasps, grasp_conf = GraspGenSampler.run_inference(
                point_cloud,
                self.sampler,
                num_grasps=num_grasps,
                topk_num_grasps=topk_num_grasps,
                grasp_threshold=grasp_threshold,
                min_grasps=min_grasps,
                max_tries=max_tries,
                remove_outliers=remove_outliers,
            )

            generation_time = time.time() - start_time

            # Convert to numpy
            if len(grasps) > 0:
                grasps_np = grasps.cpu().numpy().astype(np.float32)
                conf_np = grasp_conf.cpu().numpy().astype(np.float32)
            else:
                grasps_np = np.empty((0, 4, 4), dtype=np.float32)
                conf_np = np.empty((0,), dtype=np.float32)

            logger.info(
                f"Generated {len(grasps_np)} grasps in {generation_time:.2f}s "
                f"(conf range: {conf_np.min():.3f} - {conf_np.max():.3f})"
                if len(conf_np) > 0
                else ""
            )

            return {
                "grasps": grasps_np,
                "confidences": conf_np,
                "metadata": {
                    "num_grasps": len(grasps_np),
                    "generation_time": round(generation_time, 2),
                    "num_points": len(point_cloud),
                    "gripper_name": self.gripper_name,
                    "model_name": self.model_name,
                },
            }

        except Exception as e:
            logger.error(f"Grasp generation failed: {e}")
            raise RuntimeError(f"Grasp generation failed: {e}") from e

    def _create_success_response(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a success response."""
        return {
            "status": "success",
            "grasps": self._encode_numpy_array(results["grasps"]),
            "confidences": self._encode_numpy_array(results["confidences"]),
            "metadata": results["metadata"],
        }

    def _create_error_response(
        self, error: str, error_type: str = "RuntimeError"
    ) -> Dict[str, Any]:
        """Create an error response."""
        return {"status": "error", "error": error, "error_type": error_type}

    def handle_request(self, request_json: str) -> str:
        """
        Process a single JSON request.

        Args:
            request_json: JSON string containing the request

        Returns:
            JSON string containing the response
        """
        try:
            request = json.loads(request_json)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON request: {e}")
            return json.dumps(
                self._create_error_response(f"Invalid JSON: {e}", "JSONDecodeError")
            )

        try:
            self._validate_request(request)
        except ValueError as e:
            logger.error(f"Invalid request: {e}")
            return json.dumps(self._create_error_response(str(e), "ValueError"))

        try:
            # Decode point cloud
            point_cloud = self._decode_numpy_array(request["point_cloud"])
            point_cloud = point_cloud.astype(np.float32) # Ensure correct dtype
            assert point_cloud.ndim == 2 and point_cloud.shape[1] == 3, "Point cloud must be (N, 3)"

            if point_cloud.shape[0] < 10:
                return json.dumps(
                    self._create_error_response(
                        "Point cloud must have at least 10 points", "ValueError"
                    )
                )

            # Get options
            options = request.get("options", {})

            # Override with top-level parameters if provided
            if "num_grasps" in request:
                options["num_grasps"] = request["num_grasps"]
            if "topk_num_grasps" in request:
                options["topk_num_grasps"] = request["topk_num_grasps"]
            if "grasp_threshold" in request:
                options["grasp_threshold"] = request["grasp_threshold"]

            # Generate grasps
            results = self._generate_grasps(point_cloud, options)
            return json.dumps(self._create_success_response(results))

        except RuntimeError as e:
            logger.error(f"Generation error: {e}")
            return json.dumps(self._create_error_response(str(e), "RuntimeError"))

        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}")
            return json.dumps(
                self._create_error_response(
                    f"Internal server error: {e}", "InternalServerError"
                )
            )

    def start(self) -> None:
        """
        Start the ZMQ server and begin processing requests.

        This method blocks until the server is stopped.
        """
        logger.info("Starting GraspGen ZMQ server...")

        # Load model before starting server
        if self.sampler is None:
            self.load_model()

        # Create ZMQ context and REP socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout)

        bind_address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(bind_address)
        logger.info(f"Server bound to {bind_address}")
        logger.info("Server ready to receive requests...")

        self.request_count = 0

        try:
            while True:
                try:
                    request_json = self.socket.recv_string()
                    self.request_count += 1
                    logger.info(f"Received request #{self.request_count}")

                    response_json = self.handle_request(request_json)
                    self.socket.send_string(response_json)

                except zmq.Again:
                    # Timeout - sleep briefly to reduce CPU usage
                    time.sleep(0.1)
                    continue

                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
                    break

                except Exception as e:
                    logger.error(f"Error in request loop: {e}")

                    error_response = json.dumps(
                        self._create_error_response(f"Server error: {e}", "ServerError")
                    )
                    self.socket.send_string(error_response)

        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the ZMQ server and cleanup resources."""
        logger.info("Stopping GraspGen ZMQ server...")

        if self.socket:
            self.socket.close()
            logger.info("ZMQ socket closed")

        if self.context:
            self.context.term()
            logger.info("ZMQ context terminated")

        logger.info(f"Server stopped. Total requests processed: {self.request_count}")


def main():
    """
    Main entry point for the GraspGen ZMQ server.

    Parses command-line arguments, initializes the server, and starts
    processing requests.
    """
    parser = argparse.ArgumentParser(
        description="GraspGen ZMQ Server - 6-DOF Grasp Generation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--host",
        type=str,
        default=DEFAULT_HOST,
        help="Host to bind to (* for all interfaces)",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT, help="ZMQ port to listen on"
    )
    parser.add_argument("--gpu", type=int, default=DEFAULT_GPU_ID, help="GPU device ID")
    parser.add_argument(
        "--recv-timeout",
        type=int,
        default=DEFAULT_RECV_TIMEOUT,
        help="ZMQ receive timeout in milliseconds",
    )
    parser.add_argument(
        "--gripper-config",
        type=str,
        default=DEFAULT_GRIPPER_CONFIG,
        help="Gripper configuration file",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level))

    # Print configuration
    logger.info("=" * 60)
    logger.info("GraspGen ZMQ Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Gripper Config: {args.gripper_config}")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)

    server = GraspGenZMQServer(
        host=args.host,
        port=args.port,
        gpu_id=args.gpu,
        recv_timeout=args.recv_timeout,
        gripper_config=args.gripper_config,
    )

    try:
        server.start()
    except KeyboardInterrupt:
        logger.info("Server interrupted by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
