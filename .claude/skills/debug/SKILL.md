# Debug Skill
1. First check environment: API keys in .env, Python version, network connectivity
2. Run `pytest -x` to identify failing tests
3. Read error tracebacks carefully before diagnosing
4. Check for module-level side effects (lazy init pattern)
5. Verify the actual root cause before suggesting fixes
