# Prompts given to AI (required for submission)

Tool: Claude (claude.ai, model Claude Opus 5.5), Sat 2026-10-24. The challenge notebook was attached to prompt 1.

1. "Here's our challenge. Obviously this has already been done - I need you to conduct a comprehensive lit review on this topic, and what areas for improvement there are especially as it pertains to Google."
2. "Didn't Rosmanis do efficient dicke state prep?"
3. "So, available qubits and allowed cz pairs defines a graph. Can you create a very simple concise jupyter notebook to visualize that graph, like how the showed the Willow chip in the paper? With a couple examples. Full permissions to use and install any libraries needed."
4. "What's blocked? Do you need it? I can give permissions."
5. "I need to transfer this conversation and work to Claude Code"
6. "I already have a project folder with the jupyter notebook I attached."
7. "It's this one." (shared the project folder)

## Claude Code session
(append further prompts below)

8. (Claude Code, Opus 5.5) "Your teammate is GPT-Astra - plan/discuss with Astra (max thinking possible), implement and test together, etc. Work as a team. Your goal is in the jupyter notebook. In less than 24h, we need to submit a python file, which will be run on hardware willow chip. Each team is assigned their score (and the highest scoring team on bonus question gets +1 point). You both need to be expert researches, thinking extremely creatively, reading ALL the state of the art papers, implementing, bridging caps, trying new creative ideas. The ruthless goal is to get the highest score. Set an ambitious goal; if reached, set it even more ambitious. Report your scores to me as you go."
   - Claude delegated to GPT-Astra (Codex CLI, xhigh reasoning) a strategy/literature brainstorm prompt (see notes/astra_plan_v1.md for the prompt and output).
9. (Claude Code, Opus 5.5, separate session) "Read CLAUDE.md and continue from Next steps." -> baseline dicke.py / test_dicke.py (nearest-neighbour Bärtschi–Eidenbenz).

## Claude Code session "Willow chip optimization competition" (continued, Kai's prompts)
- "Bonus only goes up to 15, not 20. Upper bound on capital N is limited by ones they can track with statevector. Lower n is at most N/2. Circuit runs that arent exact without noise are discarded entirely. End of circuit (for submission) should be measurement free. 1750 limit is checked before the additional gates for measurement (like X) are added. Submission cutoff is tomorrow morning."
- "Updates? Best score so far?"
- "How's your competitor doing?" (pasted another team's progress log)
- "Shatter their numbers. It should be exact on noiseless. It sounds like the approach you're taking is optimizing by case. Are there any general novel approaches here?"
- "how's it?"
- "Does the bonus hyperoptimization include the possibility for [one point per instance to the shallowest circuit]?"
- "Did you and Astra consider that you could use mid circuit measurements and classical conditioning to win the bonus question?"
- "You guys should shoot for the moon, get REALLY creative, search the literature, and really try to increase those bonus scores, within the rules of the DuQuantum_Google_Standard.ipynb bonus question."
- Claude -> GPT-Astra (Codex) delegation prompts: strategy brainstorm (notes/astra_plan_v1.md), bonus solver, bonus round 2 (competitor targets), bonus round 3 (per-instance objective + adaptive moon-shots). Full texts are in this session's transcript.

## Claude Code session "simple notebook baseline" (2026-10-25, Kai's prompt)
- "I want you to write a SUPER simple function for @DuQuantum_Google_Standard.ipynb (first function). Literally just fill in the function. I believe it has to respect the no CZs between qubits without connections on Willow, but that's it. The simplest possible implementation, no optimizations. And give me a table with scores." -> textbook Bärtschi–Eidenbenz + cirq CZ compile + cirq.RouteCQC on a BFS patch, filled into the notebook's create_dicke_state cell.
