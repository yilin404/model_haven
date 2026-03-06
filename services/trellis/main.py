#!/usr/bin/env python3
"""
TRELLIS ZMQ Server - Text-to-3D Generation Service

A lightweight ZMQ-based server that wraps Microsoft's TRELLIS text-to-3D
generation model, enabling remote clients to generate 3D models from text prompts.

Author: Claude Code
Date: 2025-03-05
"""

import argparse
import base64
import json
import logging
import os
import sys
import time
from io import BytesIO
from typing import Any, Dict, Optional

import torch
import zmq
from trellis.pipelines import TrellisTextTo3DPipeline
from trellis.utils import postprocessing_utils

# Set required environment variables for TRELLIS
os.environ['SPCONV_ALGO'] = 'native'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('trellis_server.log')
    ]
)
logger = logging.getLogger(__name__)


class TrellisZMQServer:
    """
    ZMQ-based server for TRELLIS text-to-3D generation.

    This server listens for JSON requests containing text prompts and
    returns base64-encoded GLB files containing the generated 3D models.
    """

    def __init__(
        self,
        host: str = "*",
        port: int = 5555,
        model_name: str = "microsoft/TRELLIS-text-xlarge",
        gpu_id: int = 0,
        timeout: int = 120,
        recv_timeout: int = 5000
    ):
        """
        Initialize the TRELLIS ZMQ server.

        Args:
            host: Host to bind to (default: "*" for all interfaces)
            port: ZMQ port to listen on (default: 5555)
            model_name: HuggingFace model identifier (default: "microsoft/TRELLIS-text-xlarge")
            gpu_id: GPU device ID (default: 0)
            timeout: Default generation timeout in seconds (default: 120)
            recv_timeout: ZMQ receive timeout in milliseconds (default: 5000)
        """
        self.host = host
        self.port = port
        self.model_name = model_name
        self.gpu_id = gpu_id
        self.timeout = timeout
        self.recv_timeout = recv_timeout

        # Initialize ZMQ context and socket
        self.context = None
        self.socket = None

        # TRELLIS pipeline (loaded on startup)
        self.pipeline = None

        # Request counter
        self.request_count = 0

        logger.info(f"TrellisZMQServer initialized: host={host}, port={port}, model={model_name}, gpu={gpu_id}")

    def _setup_cuda_device(self) -> None:
        """Configure CUDA device based on configuration."""
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            if self.gpu_id >= device_count:
                logger.warning(
                    f"Requested GPU {self.gpu_id} not available. "
                    f"Available GPUs: {device_count}. Using GPU 0."
                )
                self.gpu_id = 0
            torch.cuda.set_device(self.gpu_id)
            logger.info(f"Using CUDA device: {torch.cuda.get_device_name(self.gpu_id)}")
        else:
            logger.warning("CUDA not available. Running on CPU.")
            self.gpu_id = -1

    def load_pipeline(self) -> None:
        """
        Load the TRELLIS text-to-3D pipeline.

        This should be called once during initialization to load the model
        into GPU memory. The pipeline is cached for subsequent requests.
        """
        logger.info(f"Loading TRELLIS pipeline: {self.model_name}")
        start_time = time.time()

        try:
            self._setup_cuda_device()

            # Load pipeline from HuggingFace
            self.pipeline = TrellisTextTo3DPipeline.from_pretrained(self.model_name)

            # Move to GPU if available
            if self.gpu_id >= 0:
                self.pipeline.cuda()

            load_time = time.time() - start_time
            logger.info(f"TRELLIS pipeline loaded successfully in {load_time:.2f}s")

        except Exception as e:
            logger.error(f"Failed to load TRELLIS pipeline: {e}")
            raise RuntimeError(f"Pipeline loading failed: {e}") from e

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
        if "text" not in request:
            raise ValueError("Missing required field: 'text'")

        if not isinstance(request["text"], str) or not request["text"].strip():
            raise ValueError("Field 'text' must be a non-empty string")

        # Validate optional fields
        if "seed" in request and not isinstance(request["seed"], int):
            raise ValueError("Field 'seed' must be an integer")

        if "options" in request and not isinstance(request["options"], dict):
            raise ValueError("Field 'options' must be an object")

        # Validate options
        if "options" in request:
            options = request["options"]

            if "simplify" in options:
                simplify = options["simplify"]
                if not isinstance(simplify, (int, float)) or not (0 <= simplify <= 1):
                    raise ValueError("Option 'simplify' must be a float between 0 and 1")

            if "texture_size" in options:
                texture_size = options["texture_size"]
                if not isinstance(texture_size, int) or texture_size < 1:
                    raise ValueError("Option 'texture_size' must be a positive integer")

            if "timeout" in options:
                timeout = options["timeout"]
                if not isinstance(timeout, int) or timeout < 1:
                    raise ValueError("Option 'timeout' must be a positive integer")

    def _generate_3d_model(
        self,
        text: str,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> tuple[bytes, Dict[str, Any]]:
        """
        Generate a 3D model from text prompt using TRELLIS.

        Args:
            text: Text prompt for 3D generation
            seed: Random seed for generation (optional)
            options: Generation parameters (optional)

        Returns:
            Tuple of (glb_bytes, metadata)

        Raises:
            RuntimeError: If generation fails
        """
        if options is None:
            options = {}

        # Extract options with defaults
        simplify = options.get("simplify", 0.95)
        texture_size = options.get("texture_size", 1024)

        logger.info(f"Generating 3D model: '{text}' (seed={seed}, simplify={simplify}, texture_size={texture_size})")

        start_time = time.time()

        try:
            # Run TRELLIS pipeline
            generation_params = {"seed": seed if seed is not None else torch.randint(0, 2**32, (1,)).item()}

            # Add optional pipeline parameters if provided
            if "sparse_structure_sampler_params" in options:
                generation_params["sparse_structure_sampler_params"] = options["sparse_structure_sampler_params"]
            if "slat_sampler_params" in options:
                generation_params["slat_sampler_params"] = options["slat_sampler_params"]

            outputs = self.pipeline.run(text, **generation_params)

            # Extract components
            gaussian = outputs['gaussian'][0]
            mesh = outputs['mesh'][0]

            logger.info("Post-processing and exporting to GLB format")

            # Convert to GLB using TRELLIS postprocessing utilities
            glb_mesh = postprocessing_utils.to_glb(
                gaussian,
                mesh,
                simplify=simplify,
                texture_size=texture_size,
                fill_holes=True,
                verbose=False
            )

            # Export GLB to bytes
            buffer = BytesIO()
            glb_mesh.export(file_obj=buffer, file_type='glb')
            glb_bytes = buffer.getvalue()

            generation_time = time.time() - start_time

            metadata = {
                "prompt": text,
                "seed": generation_params["seed"],
                "generation_time": round(generation_time, 2),
                "file_size": len(glb_bytes),
                "simplify": simplify,
                "texture_size": texture_size
            }

            logger.info(f"Generation complete in {generation_time:.2f}s, file size: {len(glb_bytes)} bytes")

            return glb_bytes, metadata

        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            raise RuntimeError(f"Generation failed: {e}") from e

    def _create_success_response(self, glb_data: bytes, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a success response.

        Args:
            glb_data: GLB file bytes
            metadata: Generation metadata

        Returns:
            JSON-serializable response dictionary
        """
        return {
            "status": "success",
            "glb_data": base64.b64encode(glb_data).decode('utf-8'),
            "metadata": metadata
        }

    def _create_error_response(self, error: str, error_type: str = "RuntimeError") -> Dict[str, Any]:
        """
        Create an error response.

        Args:
            error: Error message
            error_type: Error type name

        Returns:
            JSON-serializable error response dictionary
        """
        return {
            "status": "error",
            "error": error,
            "error_type": error_type
        }

    def handle_request(self, request_json: str) -> str:
        """
        Process a single JSON request.

        Args:
            request_json: JSON string containing the request

        Returns:
            JSON string containing the response
        """
        try:
            # Parse JSON request
            try:
                request = json.loads(request_json)
            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON request: {e}")
                return json.dumps(self._create_error_response(
                    f"Invalid JSON: {e}", "JSONDecodeError"
                ))

            # Validate request
            try:
                self._validate_request(request)
            except ValueError as e:
                logger.error(f"Invalid request: {e}")
                return json.dumps(self._create_error_response(str(e), "ValueError"))

            # Extract parameters
            text = request["text"]
            seed = request.get("seed")
            options = request.get("options", {})

            # Generate 3D model
            try:
                glb_bytes, metadata = self._generate_3d_model(text, seed, options)
                return json.dumps(self._create_success_response(glb_bytes, metadata))

            except RuntimeError as e:
                logger.error(f"Generation error: {e}")
                return json.dumps(self._create_error_response(str(e), "RuntimeError"))

        except Exception as e:
            logger.error(f"Unexpected error handling request: {e}")
            return json.dumps(self._create_error_response(
                f"Internal server error: {e}", "InternalServerError"
            ))

    def start(self) -> None:
        """
        Start the ZMQ server and begin processing requests.

        This method blocks until the server is stopped.
        """
        logger.info("Starting TRELLIS ZMQ server...")

        # Load pipeline before starting server
        if self.pipeline is None:
            self.load_pipeline()

        # Create ZMQ context and REP socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # Set receive timeout to allow graceful shutdown
        self.socket.setsockopt(zmq.RCVTIMEO, self.recv_timeout)

        # Bind to address
        bind_address = f"tcp://{self.host}:{self.port}"
        self.socket.bind(bind_address)
        logger.info(f"Server bound to {bind_address}")
        logger.info("Server ready to receive requests...")

        self.request_count = 0

        try:
            while True:
                try:
                    # Receive request
                    request_json = self.socket.recv_string()
                    self.request_count += 1
                    logger.info(f"Received request #{self.request_count}")

                    # Handle request and send response
                    response_json = self.handle_request(request_json)
                    self.socket.send_string(response_json)

                except zmq.Again:
                    # Timeout - continue loop (allows checking for interrupts)
                    continue

                except KeyboardInterrupt:
                    logger.info("Shutdown requested by user")
                    break

                except Exception as e:
                    logger.error(f"Error in request loop: {e}")
                    # Try to send error response
                    try:
                        error_response = json.dumps(self._create_error_response(
                            f"Server error: {e}", "ServerError"
                        ))
                        self.socket.send_string(error_response)
                    except Exception:
                        logger.error("Failed to send error response")

        finally:
            self.stop()

    def stop(self) -> None:
        """
        Stop the ZMQ server and cleanup resources.
        """
        logger.info("Stopping TRELLIS ZMQ server...")

        if self.socket:
            self.socket.close()
            logger.info("ZMQ socket closed")

        if self.context:
            self.context.term()
            logger.info("ZMQ context terminated")

        logger.info(f"Server stopped. Total requests processed: {self.request_count}")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments and environment variables.

    Environment variables (with defaults):
        TRELLIS_ZMQ_PORT: Port (default: 5555)
        TRELLIS_MODEL: Model name (default: microsoft/TRELLIS-text-xlarge)
        TRELLIS_GPU: GPU ID (default: 0)
        TRELLIS_TIMEOUT: Timeout in seconds (default: 120)
        TRELLIS_HOST: Bind host (default: *)

    Command-line arguments override environment variables.

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="TRELLIS ZMQ Server - Text-to-3D Generation Service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("TRELLIS_ZMQ_PORT", 5555)),
        help="ZMQ port to listen on"
    )

    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("TRELLIS_HOST", "*"),
        help="Host to bind to (* for all interfaces)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("TRELLIS_MODEL", "microsoft/TRELLIS-text-xlarge"),
        help="HuggingFace model identifier"
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=int(os.environ.get("TRELLIS_GPU", 0)),
        help="GPU device ID (-1 for CPU)"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.environ.get("TRELLIS_TIMEOUT", 120)),
        help="Default generation timeout in seconds"
    )

    parser.add_argument(
        "--recv-timeout",
        type=int,
        default=5000,
        help="ZMQ receive timeout in milliseconds"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=os.environ.get("TRELLIS_LOG_LEVEL", "INFO"),
        help="Logging level"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the TRELLIS ZMQ server.

    Parses command-line arguments, initializes the server, and starts
    processing requests.
    """
    # Parse arguments
    args = parse_arguments()

    # Configure logging level
    logger.setLevel(getattr(logging, args.log_level))

    # Print configuration
    logger.info("=" * 60)
    logger.info("TRELLIS ZMQ Server Configuration")
    logger.info("=" * 60)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Model: {args.model}")
    logger.info(f"GPU: {args.gpu}")
    logger.info(f"Timeout: {args.timeout}s")
    logger.info(f"Log Level: {args.log_level}")
    logger.info("=" * 60)

    # Create and start server
    server = TrellisZMQServer(
        host=args.host,
        port=args.port,
        model_name=args.model,
        gpu_id=args.gpu,
        timeout=args.timeout,
        recv_timeout=args.recv_timeout
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
