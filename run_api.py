#!/usr/bin/env python3
"""
Script chạy API server với cấu hình từ .env
"""

import os
import uvicorn
from pathlib import Path

def load_env():
    """Load environment variables từ .env file"""
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        print(f"📄 Loading config from: {env_file}")
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
    else:
        print("⚠️ .env file not found, using default values")

def main():
    """Main function to run the API server"""
    
    # Load environment variables
    load_env()
    
    # Get configuration from environment
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    reload = os.getenv("API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("🚀 STARTING TRAFFIC AI API SERVER")
    print("=" * 50)
    print(f"📡 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"📊 Log Level: {log_level}")
    print(f"🌐 CORS Origins: {os.getenv('AI_ALLOWED_ORIGINS', '*')}")
    print(f"🔐 Auth Required: {os.getenv('REQUIRE_AUTH', 'false')}")
    print(f"📏 Max File Size: {os.getenv('MAX_FILE_SIZE_MB', '10')}MB")
    print("=" * 50)
    print(f"🌍 API Documentation: http://{host}:{port}/docs")
    print(f"📖 ReDoc: http://{host}:{port}/redoc")
    print("=" * 50)
    
    # Run the server
    uvicorn.run(
        "src.ai_service.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main()