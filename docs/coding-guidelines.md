# Coding Guidelines

Based on Clean Code by Robert C. Martin, design patterns, and production engineering best practices.

## Conventions

- **Always**: All code must adhere to these.
- **Never**: All code must never do these.
- **Usually applicable**: Most code should follow; exceptions exist.

## SOLID Principles

- **S — Single Responsibility**: A class must have one and only one reason to change. [Always]
- **O — Open/Closed**: Classes should be open for extension, closed for modification. [Always]
- **L — Liskov Substitution**: Subtypes must be substitutable for their base types without altering correctness. [Always]
- **I — Interface Segregation**: Prefer small, focused interfaces over large ones. [Always]
- **D — Dependency Inversion**: Depend on abstractions, not concretions. [Always]

## Design Principles

- **Composition over Inheritance**: Favor object composition for code reuse; inheritance creates tight coupling. [Always]
- **Least Surprise**: Code should behave as readers expect from its name and signature. [Always]
- **Fail Fast**: Validate inputs at boundaries; throw early rather than propagating invalid state. [Always]
- **Defensive Coding**: Assume external inputs are malformed. Validate, sanitize, and bound-check. [Always]
- **Separation of Concerns**: Keep business logic, I/O, and presentation in distinct layers. [Always]
- **DRY (Don't Repeat Yourself)**: Extract shared logic into reusable components. But prefer duplication over wrong abstraction. [Usually]
- **YAGNI (You Aren't Gonna Need It)**: Don't build features or abstractions speculatively. [Always]
- **Tell, Don't Ask**: Objects should perform actions, not expose state for others to act on. [Usually]
- **Law of Demeter**: A method should only call methods on its direct collaborators. [Always]

## Error Handling & Resilience

- Never swallow exceptions silently — log with context or rethrow. [Always]
- Use specific exception types, not generic `Exception`/`Error`. [Always]
- Wrap external service calls with timeouts, retries (with backoff), and circuit breakers. [Always]
- Return empty collections instead of null for collection-typed returns. [Always]
- Distinguish recoverable errors (retry/fallback) from unrecoverable ones (fail fast). [Always]
- Include correlation IDs in error logs for distributed tracing. [Usually]
- Graceful degradation: if a non-critical dependency fails, degrade the feature, don't crash the request. [Always]

## Concurrency & Thread Safety

- Prefer immutable objects — they are inherently thread-safe. [Always]
- Minimize shared mutable state. If unavoidable, synchronize access and document the contract. [Always]
- Use thread-safe collections or synchronization primitives, never raw mutable collections across threads. [Always]
- Avoid holding locks during I/O or network calls. [Always]
- Design for idempotency in distributed operations. [Always]

## Performance & Scalability

- Avoid premature optimization — profile first, optimize bottlenecks. [Usually]
- Cache expensive computations; define TTL and invalidation strategy. [Usually]
- Use lazy initialization for heavy resources not always needed. [Usually]
- Prefer streaming/pagination over loading unbounded data into memory. [Always]
- Bound all collections, queues, and buffers — unbounded growth causes OOMs in production. [Always]
- Avoid blocking the main/UI thread with I/O or computation. [Always]

## API & Interface Design

- APIs should be hard to misuse and easy to use correctly. [Always]
- Use builder pattern or named parameters when constructors exceed 3 arguments. [Usually]
- Make illegal states unrepresentable through types (enums, sealed classes, non-null types). [Always]
- Version public APIs; never make breaking changes without a migration path. [Always]
- Document preconditions, postconditions, and side effects in public API contracts. [Always]

## Observability & Operations

- Emit metrics for business-critical operations (latency, error rate, throughput). [Always]
- Log at appropriate levels: ERROR for actionable failures, WARN for degraded state, DEBUG for diagnostics. [Always]
- Never log sensitive data (PII, tokens, passwords, full credit card numbers). [Always]
- Include enough context in logs to diagnose production issues without a debugger. [Always]
- Every new feature behind a weblab/feature flag must be rollback-safe without a deployment. [Always]

## Comments

- **C1**: No inappropriate info in comments (author, dates — use VCS). [Always]
- **C3**: No redundant comments — code should speak for itself. [Always]
- **C4**: Use correct grammar, be brief, don't state the obvious. [Always]
- **C5**: Delete commented-out code — VCS remembers it. [Always]

## Functions

- **F1**: Minimize arguments (0 best, then 1, 2, 3 max). [Usually]
- **F2**: No output arguments passed as parameters. [Always]
- **F4**: Delete dead/uncalled functions. [Always]
- **F5**: Never pass or return null — throw exceptions or use Optional. [Always]
- **F7**: Minimize side effects; write pure functions. Name side-effecting functions clearly. [Always]

## General

- **G5**: Avoid duplication (code + config). [Usually]
- **G5a**: Always research existing code references before implementing new integrations. Find a working example in the codebase first, then follow the same pattern. Never guess API usage. [Always]
- **G9**: Remove dead code (unreachable if/catch/switch). [Always]
- **G11**: Be consistent — same patterns for same things. [Always]
- **G12**: Remove clutter (unused vars, empty constructors). [Always]
- **G16**: No obscured intent — no magic numbers, no Hungarian notation. [Always]
- **G20**: Function names must say what they do. [Always]
- **G24**: Follow standard conventions (formatting, naming). [Always]
- **G25**: Replace magic numbers with named constants. [Always]
- **G30**: Functions should do one thing. [Always]
- **G36**: No transitive navigation (a.getB().getC().doSomething()). [Always]
- **G37**: Prefer immutable collections. Document why if mutable. [Always]

## Names

- **N1**: Choose descriptive names. [Always]
- **N4**: Unambiguous names. [Always]
- **N6**: No encodings (m_, f prefixes). [Always]
- **N7**: Names should describe side-effects. [Always]

## Java/Kotlin Specific

- **J1**: Never use wildcard imports. [Never]
- **J2**: Don't inherit constants — use static/regular imports. [Always]
- **J3**: Use enums over public static final ints. [Always]

## Tests

- **T0**: Every new class, method, or behavioral change must ship with unit tests in the same CR. No exceptions. [Always]
- **T1**: Maintain ≥85% line coverage and ≥80% branch coverage. [Usually]
- **T5**: Test boundary conditions. [Always]
- **T6**: Exhaustively test near bugs. [Always]
- **T9**: Tests should be fast. [Usually]
- **T10**: Single concept per test. [Always]
- **T11**: Tests should be precise — avoid anyThing/isA matchers. [Usually]

## Dependencies & Security

- Pin dependency versions — no open ranges. [Always]
- Prefer well-maintained, widely-used packages over obscure alternatives. [Always]
- Validate and sanitize all external inputs — never trust client data. [Always]
- Use parameterized queries — never concatenate user input into queries. [Always]
- Use short-lived credentials; never hardcode secrets in source. [Always]
