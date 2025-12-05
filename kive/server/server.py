"""
Kive Server implementation

Provides FastAPI service that exposes HTTP interface for the memory system
"""

import os
from typing import Optional
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api.routes import router
from .cache import MemoCache
from ..adapters.base import BaseMemoryAdapter
from ..utils.logger import logger, LOG_LEVEL


class Server:
    """Kive Server
    
    Example:
        engine = Cognee(...)
        server = Server(engine=engine, port=8000)
        server.run()
    """
    
    def __init__(
        self,
        engine: BaseMemoryAdapter,
        host: str = "0.0.0.0",
        port: int = 12306,
        log_level: Optional[str] = None,
        enable_cors: bool = True,
        cache_dir: str = ".kive/memo_cache",
    ):
        """
        Args:
            engine: Memory engine
            host: Listen address
            port: Listen port
            log_level: Log level, overrides environment variable
            enable_cors: Enable CORS
            cache_dir: Memo cache directory
        """
        self.adapter = engine  # Store as adapter internally
        self.host = host
        self.port = port
        self.enable_cors = enable_cors
        self.cache_dir = cache_dir
        
        # Update log level
        if log_level:
            self._update_log_level(log_level)
        
        # Create memo cache
        self.cache = MemoCache(cache_dir)
        
        # Create FastAPI app
        self.app = self._create_app()
    
    def _update_log_level(self, log_level: str):
        """Update global log level"""
        import os
        import sys
        os.environ["KIVE_LOG_LEVEL"] = log_level.upper()
        
        # Update loguru logger level
        from loguru import logger as _logger
        _logger.remove()  # Remove default handler
        _logger.add(
            sink=sys.stderr,
            level=log_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
            colorize=True
        )
        
        logger.info(f"Log level updated to: {log_level}")
    
    def _create_app(self) -> FastAPI:
        """创建FastAPI应用"""
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            logger.info("Kive server starting...")
            
            # Initialize backend
            await self.adapter.initialize()
            
            # Start background processor
            await self.adapter.start_background_processor()
            
            logger.info(
                f"Kive server started on http://{self.host}:{self.port}"
            )
            
            yield
            
            # Shutdown
            logger.info("Kive server shutting down...")
            await self.adapter.close()
            self.cache.close()
            logger.info("Kive server stopped")
        
        app = FastAPI(
            title="Kive Memory System",
            description="A lightweight memory system",
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
            lifespan=lifespan
        )
        
        # 添加CORS中间件
        if self.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
        
        # Register routes
        app.include_router(router)
        
        # Store backend and cache to app state
        app.state.adapter = self.adapter  # Internal uses 'adapter'
        app.state.cache = self.cache
        
        return app
    
    def run(self):
        """启动服务器(阻塞式)"""
        logger.info(f"Starting Kive server on {self.host}:{self.port}")
        
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level=LOG_LEVEL.lower()
        )
