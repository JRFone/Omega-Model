# Omega native libraries

This directory receives the platform-specific shared library produced by:

```text
python build_native_backend.py --clean
```

The Python application automatically loads:

- `omega_native.dll` on Windows;
- `libomega_native.so` on Linux;
- `libomega_native.dylib` on macOS.

If no compatible library is present, Omega retains a transparent Python fallback
and reports that the accelerated backend is unavailable. Scientific results must
not change when the backend changes; parity tests enforce this requirement.
