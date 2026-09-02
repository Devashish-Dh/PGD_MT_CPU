# Parallel gradient descent on CPU

Multi-threaded gradient descent on simple functions, used to practice **races, atomics, and the surrounding tooling** (CMake, warnings, optional static analysis).

Untuned run notes: [`untuned_obs.txt`](untuned_obs.txt)

## Build

C++17, pthreads via CMake `Threads`.

```bash
cmake -S . -B build
cmake --build build
```

Sources: [`src/`](src/) · headers: [`include/`](include/) · [`CMakeLists.txt`](CMakeLists.txt)
