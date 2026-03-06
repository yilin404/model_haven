# TRELLIS ZMQ Server

A lightweight ZMQ-based server that wraps Microsoft's TRELLIS text-to-3D generation model, enabling remote clients to generate 3D models from text prompts.

## Overview

This server provides a simple request/response API for generating 3D assets from natural language descriptions using the TRELLIS model. It uses ZeroMQ (ZMQ) for efficient communication and handles the heavy lifting of 3D generation, mesh processing, and GLB export.

**Why use this server?**

- **Remote 3D Generation**: Offload GPU-intensive 3D generation to a dedicated server
- **Simple API**: Send text prompts, receive GLB 3D models
- **Resource Management**: TRELLIS-xlarge requires ~8GB GPU memory - this server manages GPU resources efficiently
- **Language Agnostic**: ZMQ supports clients in any programming language
- **Request Queuing**: ZMQ handles multiple concurrent clients automatically

## Features

- Text-to-3D generation using Microsoft TRELLIS
- Automatic mesh simplification and texturing
- GLB format output (ready for web/mobile/AR)
- Configurable generation parameters
- Error handling and timeout management
- Multi-GPU support (run multiple instances)

## Installation

### Prerequisites

- Python 3.11+
- CUDA-capable GPU with at least 8GB VRAM
- CUDA Toolkit (compatible with your PyTorch version)

### Step 1: Create Virtual Environment

```bash
cd model_haven/services/trellis
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install with pip (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install zmq transformers trimesh open3d plyfile imageio pillow

# Or install with uv (faster)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install -e .
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "import zmq; print('ZMQ installed successfully')"
```

## Quick Start

### Start the Server

```bash
# Basic usage (default port 5555, GPU 0)
python main.py

# Custom configuration
python main.py --port 6000 --gpu 1

# With environment variables
export TRELLIS_ZMQ_PORT=5555
export TRELLIS_GPU=0
python main.py
```

You should see output like:

```
Loading TRELLIS model: microsoft/TRELLIS-text-xlarge
Model loaded on device: cuda:0
ZMQ server started on tcp://*:5555
Waiting for requests...
```

### Test with Python Client

Create a simple test client:

```python
import zmq
import json
import base64

# Connect to server
context = zmq.Context()
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

# Send request
request = {
    "text": "A modern chair with wooden legs",
    "seed": 42
}
socket.send_json(request)

# Receive response
response = socket.recv_json()

if response["status"] == "success":
    # Save GLB file
    glb_data = base64.b64decode(response["glb_data"])
    with open("chair.glb", "wb") as f:
        f.write(glb_data)
    print(f"Generated in {response['metadata']['generation_time']:.2f}s")
    print(f"File size: {response['metadata']['file_size']} bytes")
else:
    print(f"Error: {response['error']}")
```

## Usage

### Basic Request

```python
request = {
    "text": "A red coffee mug"
}
```

### Request with Options

```python
request = {
    "text": "A wooden table with four legs",
    "seed": 123,  # Reproducible generation
    "options": {
        "simplify": 0.95,      # Mesh simplification (0-1, higher = simpler)
        "texture_size": 1024,  # Texture resolution (512, 1024, 2048)
        "timeout": 120         # Max generation time in seconds
    }
}
```

### Multiple Requests (Sequential)

```python
prompts = [
    "A blue vase",
    "A wooden stool",
    "A metal lamp"
]

for prompt in prompts:
    request = {"text": prompt}
    socket.send_json(request)
    response = socket.recv_json()

    if response["status"] == "success":
        glb_data = base64.b64decode(response["glb_data"])
        filename = f"{prompt.replace(' ', '_')}.glb"
        with open(filename, "wb") as f:
            f.write(glb_data)
        print(f"Generated {filename}")
```

### Error Handling

```python
try:
    request = {"text": "A futuristic spaceship"}
    socket.send_json(request)
    response = socket.recv_json()

    if response["status"] == "success":
        # Process successful response
        glb_data = base64.b64decode(response["glb_data"])
    else:
        # Handle server error
        print(f"Server error ({response['error_type']}): {response['error']}")

except zmq.ZMQError as e:
    print(f"Communication error: {e}")

except Exception as e:
    print(f"Unexpected error: {e}")
```

## API Reference

### Request Format

All requests must be JSON objects with the following structure:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | Yes | Text prompt for 3D generation |
| `seed` | integer | No | Random seed for reproducibility (default: random) |
| `options` | object | No | Generation parameters (see below) |

#### Options Object

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `simplify` | float | 0.95 | Mesh simplification ratio (0.0-1.0) |
| `texture_size` | integer | 1024 | Texture resolution in pixels |
| `timeout` | integer | 120 | Maximum generation time in seconds |

### Response Format

#### Success Response

```json
{
    "status": "success",
    "glb_data": "base64-encoded GLB file",
    "metadata": {
        "prompt": "A modern chair",
        "seed": 42,
        "generation_time": 45.2,
        "file_size": 1048576
    }
}
```

#### Error Response

```json
{
    "status": "error",
    "error": "Invalid parameter value: simplify must be between 0 and 1",
    "error_type": "ValueError"
}
```

### Error Types

| Error Type | Description |
|------------|-------------|
| `ValueError` | Invalid input parameters |
| `RuntimeError` | TRELLIS generation failure |
| `TimeoutError` | Generation exceeded timeout |
| `GPUMemoryError` | Insufficient GPU memory |
| `ConnectionError` | ZMQ communication error |

## Configuration

### Environment Variables

Configure the server using environment variables:

```bash
# ZMQ Configuration
export TRELLIS_ZMQ_PORT=5555        # ZMQ port (default: 5555)
export TRELLIS_HOST=*               # Bind host (default: *)

# Model Configuration
export TRELLIS_MODEL=microsoft/TRELLIS-text-xlarge  # Model name
export TRELLIS_GPU=0                # GPU device ID (default: 0)

# Generation Configuration
export TRELLIS_TIMEOUT=120          # Default timeout in seconds (default: 120)
```

### Command-Line Arguments

Override environment variables with command-line arguments:

```bash
python main.py [OPTIONS]

Options:
  --port INTEGER          ZMQ port number [default: 5555]
  --host TEXT             Bind host address [default: *]
  --model TEXT            Model name [default: microsoft/TRELLIS-text-xlarge]
  --gpu INTEGER           GPU device ID [default: 0]
  --timeout INTEGER       Default timeout in seconds [default: 120]
  --verbose               Enable verbose logging
  --help                  Show help message
```

### Configuration File (Optional)

Create a `.env` file in the service directory:

```bash
# .env
TRELLIS_ZMQ_PORT=5555
TRELLIS_GPU=0
TRELLIS_TIMEOUT=120
```

Load with:

```bash
source .env
python main.py
```

## Client Examples

### Python Client (Basic)

```python
import zmq
import json
import base64

def generate_3d(text, seed=None):
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")

    request = {"text": text}
    if seed is not None:
        request["seed"] = seed

    socket.send_json(request)
    response = socket.recv_json()

    if response["status"] == "success":
        return base64.b64decode(response["glb_data"])
    else:
        raise Exception(response["error"])

# Usage
glb_data = generate_3d("A wooden table")
with open("table.glb", "wb") as f:
    f.write(glb_data)
```

### Python Client (Advanced)

```python
import zmq
import json
import base64
from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationOptions:
    simplify: float = 0.95
    texture_size: int = 1024
    timeout: int = 120

class TrellisClient:
    def __init__(self, host="localhost", port=5555):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(f"tcp://{host}:{port}")

    def generate(
        self,
        text: str,
        seed: Optional[int] = None,
        options: Optional[GenerationOptions] = None
    ) -> tuple[bytes, dict]:
        """Generate 3D model from text prompt.

        Returns:
            (glb_data, metadata): Tuple of GLB binary data and metadata
        """
        request = {"text": text}
        if seed is not None:
            request["seed"] = seed
        if options is not None:
            request["options"] = {
                "simplify": options.simplify,
                "texture_size": options.texture_size,
                "timeout": options.timeout
            }

        self.socket.send_json(request)
        response = self.socket.recv_json()

        if response["status"] == "success":
            glb_data = base64.b64decode(response["glb_data"])
            return glb_data, response["metadata"]
        else:
            raise Exception(f"{response['error_type']}: {response['error']}")

    def close(self):
        self.socket.close()
        self.context.term()

# Usage
client = TrellisClient()
try:
    glb_data, metadata = client.generate(
        text="A futuristic chair",
        seed=42,
        options=GenerationOptions(simplify=0.98, texture_size=2048)
    )
    with open("chair.glb", "wb") as f:
        f.write(glb_data)
    print(f"Generated in {metadata['generation_time']:.2f}s")
finally:
    client.close()
```

### Node.js Client

```javascript
const zmq = require('zeromq');
const fs = require('fs');

async function generate3D(text, seed = null) {
    const sock = new zmq.Request();
    sock.connect('tcp://localhost:5555');

    const request = { text };
    if (seed !== null) request.seed = seed;

    await sock.send(JSON.stringify(request));
    const [response] = await sock.receive();

    const result = JSON.parse(response.toString());

    if (result.status === 'success') {
        const glbData = Buffer.from(result.glb_data, 'base64');
        fs.writeFileSync('output.glb', glbData);
        console.log(`Generated in ${result.metadata.generation_time}s`);
    } else {
        console.error(`Error: ${result.error}`);
    }

    sock.close();
}

// Usage
generate3D('A modern lamp', 42);
```

### C++ Client

```cpp
#include <zmq.hpp>
#include <nlohmann/json.hpp>
#include <fstream>
#include <base64.h>

nlohmann::json generate3D(const std::string& text, int seed = -1) {
    zmq::context_t context(1);
    zmq::socket_t socket(context, ZMQ_REQ);
    socket.connect("tcp://localhost:5555");

    nlohmann::json request;
    request["text"] = text;
    if (seed >= 0) request["seed"] = seed;

    std::string request_str = request.dump();
    zmq::message_t msg(request_str.begin(), request_str.end());
    socket.send(msg, zmq::send_flags::none);

    zmq::message_t response;
    socket.recv(response, zmq::recv_flags::none);

    return nlohmann::json::parse(response.to_string());
}

// Usage
int main() {
    auto result = generate3D("A wooden table", 42);

    if (result["status"] == "success") {
        std::string glb_data = base64_decode(result["glb_data"]);
        std::ofstream file("table.glb", std::ios::binary);
        file.write(glb_data.data(), glb_data.size());
        file.close();
    }
}
```

## Troubleshooting

### Common Issues

#### Issue: "CUDA out of memory"

**Cause**: TRELLIS-xlarge requires ~8GB GPU memory. Other processes may be using GPU memory.

**Solutions**:
1. Check GPU memory usage:
   ```bash
   nvidia-smi
   ```
2. Stop other GPU processes or use a different GPU:
   ```bash
   python main.py --gpu 1
   ```
3. Reduce batch size if processing multiple requests

#### Issue: "Connection refused"

**Cause**: Server is not running or wrong port/host.

**Solutions**:
1. Verify server is running:
   ```bash
   ps aux | grep "python main.py"
   ```
2. Check correct port:
   ```bash
   netstat -tuln | grep 5555
   ```
3. Verify client connection string matches server binding

#### Issue: "Timeout waiting for response"

**Cause**: Generation took longer than timeout.

**Solutions**:
1. Increase timeout in request:
   ```python
   request = {
       "text": "A complex scene",
       "options": {"timeout": 300}  # 5 minutes
   }
   ```
2. Increase default timeout:
   ```bash
   export TRELLIS_TIMEOUT=300
   python main.py
   ```

#### Issue: "Model loading failed"

**Cause**: Missing dependencies or network issues downloading model.

**Solutions**:
1. Check internet connection (first run downloads model)
2. Install missing dependencies:
   ```bash
   pip install transformers trimesh open3d
   ```
3. Clear HuggingFace cache and retry:
   ```bash
   rm -rf ~/.cache/huggingface
   python main.py
   ```

#### Issue: "ImportError: No module named 'zmq'"

**Cause**: ZMQ not installed.

**Solution**:
```bash
pip install pyzmq
```

#### Issue: Slow generation

**Cause**: TRELLIS generation is GPU-intensive (30-60 seconds normal).

**Solutions**:
1. Use a faster GPU
2. Reduce texture size:
   ```python
   request = {"text": "...", "options": {"texture_size": 512}}
   ```
3. Increase mesh simplification:
   ```python
   request = {"text": "...", "options": {"simplify": 0.98}}
   ```

### Debug Mode

Enable verbose logging:

```bash
python main.py --verbose
```

### Check Logs

Server logs include:
- Model loading status
- Each request received
- Generation progress
- Errors with stack traces

Monitor logs in real-time:

```bash
python main.py 2>&1 | tee server.log
```

## Performance

### Benchmarks

| Model | GPU | Memory | Time (avg) |
|-------|-----|--------|------------|
| TRELLIS-text-xlarge | RTX 3090 | 8GB | 30-45s |
| TRELLIS-text-xlarge | RTX 4090 | 8GB | 20-30s |
| TRELLIS-text-xlarge | A100 | 8GB | 15-25s |

### Optimization Tips

1. **Multiple GPUs**: Run separate server instances on different GPUs
   ```bash
   # Terminal 1
   python main.py --port 5555 --gpu 0 &

   # Terminal 2
   python main.py --port 5556 --gpu 1 &

   # Client: load balance across ports
   ```

2. **Reduce Texture Size**: Lower texture size for faster generation
   ```python
   request = {"text": "...", "options": {"texture_size": 512}}
   ```

3. **Increase Simplification**: Higher values produce simpler meshes
   ```python
   request = {"text": "...", "options": {"simplify": 0.98}}
   ```

4. **Use Seed**: Enable caching of common generations (future feature)

## Architecture

```
Client(s)          Server
   |                 |
   |--[JSON Request]--->|
   |                    |- Parse Request
   |                    |- Generate 3D (TRELLIS)
   |                    |- Export to GLB
   |                    |- Encode to base64
   |<-[JSON Response]----|
```

### Design Decisions

- **Single-Process REP/REQ Pattern**: Simplest implementation, ZMQ handles queuing
- **Base64 Encoding**: Language-agnostic binary data transfer
- **GLB Format**: Widely supported 3D format (web, mobile, AR)
- **Timeout Protection**: Prevents hung requests

## Future Enhancements

1. **Multi-GPU Support**: Distribute across multiple GPUs
2. **Batch Processing**: Process multiple prompts in one request
3. **Caching**: Cache common generations
4. **Pub/Sub**: Broadcast status updates for long generations
5. **Image-to-3D**: Add support for TRELLIS-image models
6. **WebSocket Support**: Enable web client connections
7. **Authentication**: Add API key authentication
8. **Rate Limiting**: Prevent abuse

## Contributing

Contributions welcome! Areas for improvement:

- Additional client examples (Go, Rust, Java)
- Performance optimizations
- Additional output formats (OBJ, STL, PLY)
- Web UI for testing

## License

This server uses the TRELLIS model from Microsoft. Please refer to the TRELLIS license for model usage terms.

## Support

For issues and questions:
1. Check this README's Troubleshooting section
2. Review server logs with `--verbose` flag
3. Open an issue on the project repository

## References

- [TRELLIS Paper](https://arxiv.org/abs/2409.17418)
- [TRELLIS GitHub](https://github.com/Microsoft/TRELLIS)
- [ZeroMQ Documentation](https://zeromq.org/)
- [GLB Format Spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html)
