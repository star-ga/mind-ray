# Pre-0.10 MIND sources (reference only)

These modules (`vec3`, `ray`, `camera`, `rng`, `material`, `hittable`, `scene`,
`render`, `io`, `backend`, `math_stubs`, `main`) are the original modular design
of the MIND renderer. They use language features that the released `mindc`
v0.10.2 does not compile (generic type aliases, tuples, `for` loops over
iterators, structs with float fields, string methods, `#[extern]` blocks), so
none of them type-check today.

The renderer that builds is `src/main.mind`. It keeps the same scenes and
materials in the executable subset of v0.10.2. These files stay here as the
layout to return to when released `mindc` supports float-field structs and f64
values across module boundaries.
