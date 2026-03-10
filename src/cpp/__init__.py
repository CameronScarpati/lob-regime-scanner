"""C++ LOB engine — optional high-performance order book reconstructor."""

try:
    from src.cpp._lob_cpp import LOBEngine, batch_reconstruct  # noqa: F401

    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
