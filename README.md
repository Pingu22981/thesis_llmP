# LLM+P: NL to PDDL translation with Llama 3.1 8B

Evaluates how well Llama 3.1 8B translates natural-language planning problems
into PDDL, scored by Planetarium semantic equivalence. Part of MSc thesis on
LLM+P for disaster-relief planning.

## Scripts
- run_experiment.py  - main runner. Flags: --domain {blocksworld,gripper},
                        --shots N, --goal-type {concrete,abstract,all},
                        --num-objects N (filter by size), --match-examples,
                        --n N (sample size), --output FILE.
- run_rewording.py   - same pipeline, swaps goal phrasing (phrasing experiment).
- run_retry.py       - same pipeline plus detect-and-retry on contradictions.
- compare.py         - reads result jsonls and prints summary tables.

## Pipeline (what run_experiment does)
1. Load Planetarium dataset (BatsResearch/planetarium).
2. Filter test set to domain + goal type (+ size if given).
3. Pick k worked examples from train split (spread across sizes).
4. Build prompt, send NL to Llama via Ollama (/api/chat, temp 0).
5. extract_pddl(): pull the (define ...) block by balancing parens.
6. planetarium.evaluate(): score parseable + equivalent vs ground truth.
7. Save one JSON line per problem.

## Key results (n=500 random unless noted)
- concrete blocksworld, small examples : 90% parseable, 19% equivalent
- concrete blocksworld, spread examples : 99% parseable, 83% equivalent
- abstract blocksworld (all 4 example configs): 0% equivalent
- Finding: concrete failure = unrepresentative examples (fixable by prompting).
  Abstract failure = reasoning limit (unmovable by prompting, tested 4 ways).

## results/ folders
- baseline_matrix/ : original 500-sample runs, 0/4 shot, concrete/abstract, both domains
- size_sweep/      : concrete blocksworld by object count (finds the breakpoint)
- spread_fix/      : re-runs with size-spanning examples (the fix)
- matched/         : abstract with matched abstract examples
- rewording/       : phrasing variants of abstract goals
- gripper/         : gripper runs
- old_superseded/  : early n=300 pilots, NOT for reporting

## Environment
Hex cluster, container 'thesis'. Ollama serves llama3.1:8b.
Domain files at /workspace/planetarium/planetarium/domains/.
