# Minimum Kullback-Leibler Estimator using Nearest Neighbours in Rust

(c) 2025, Carlos Aya-Moreno

Example implementation of the MKL in Rust / Burn.

Using [Burn](https://burn.dev/) as it seems a promising alternative to Python's torch,
although this estimator is _not_ a traditional NN model.

Implements the estimator in [paper.pdf](paper.pdf), equations (5), (6) and (7).

1. Install Rust and Cargo

2. Run `cargo build` to compile

3. Run `cargo run --example normal -- [<loc> [<scale> [<num>]]] [-s <seed>] [--split]` to run an example, e.g.

WORK IN PROGRESS:
- Errors with NaN values at the moment
