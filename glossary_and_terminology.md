# Glossary and Terminology Guide

## 1. Purpose

This file controls terminology for the Sparse Readout Prism paper. Use it to keep the paper precise, internally consistent, and grounded in the mechanistic interpretability, logit-lens, output-embedding, sparse-autoencoder, and attribution literature. It is not a list of every noun phrase in the paper. It focuses on terms that affect claims, positioning, mathematical definitions, experimental interpretation, and reviewer understanding.

## 2. Terminology Policy

- Prefer standard literature terms where they exist: use "logit lens", "unembedding matrix", "LM head", "sparse autoencoder", "decoder direction", "reconstruction error", "direct logit attribution", "baseline", and "ablation" in their usual senses.
- Use paper-specific terms only when they name a real SRP object or distinction: "Sparse Readout Prism", "readout feature", "readout feature projection", "scalar readout target", "target coefficient", "signed feature contribution", and "local score decomposition".
- Define key terms before relying on them, especially "readout", "scalar readout target", "readout feature projection", "local score decomposition", and "feature label".
- Keep "readout" narrowly scoped to the final unembedding / LM head interface. Do not use it for the whole model, a prompt, a generation, or an internal computation.
- Avoid calling hidden-state projections onto readout SAE decoder directions "feature activations". In SAE literature, "activation" usually refers to the sparse encoder output for the object on which the SAE was trained.
- Treat "faithful", "interpretable", "robust", "general", "causal", and "control" as evidence-sensitive words. Use them only with the diagnostics, scope, and caveats stated in the paper.
- Keep terminology stable across sections. Do not alternate between "query", "target", "score", "decomposition", and "explanation" unless the distinction is intentional.
- Match wording to evidence: the fixed SRP basis and selected scalar readout targets reconstruct and decompose selected linear readout scores; they do not by themselves explain arbitrary behavior, probabilities, full generations, or causal mechanisms.

### High-Priority Replacements

These are the highest-value cleanup substitutions to run before submission.

| Search for | Prefer | Reason |
|---|---|---|
| feature activation, profile activation | readout feature projection; projection coordinate | Hidden-state dot products `h^T d_i` are not SAE encoder activations. |
| query | scalar readout target; readout target | The formal object is a selected linear readout target, not a generic query. |
| explanation, explains | local score decomposition; decomposes | The fixed SRP basis plus a selected target decomposes selected linear readout scores, not arbitrary behavior. |
| benchmark performance, task accuracy | benchmark-derived readout-target fidelity | The benchmark-derived rows evaluate readout contrast reconstruction, not end-to-end task success. |
| steering, control | constrained readout-side edit; readout-level intervention | The edit experiment is score-level and candidate-constrained. |
| robust | controlled; consistent across tested settings | "Robust" needs stronger perturbation/distribution/seed evidence unless scoped to controls. |
| semantic feature | feature label; row-level descriptor | Feature labels summarize associated token rows and are not ground-truth semantics. |
| causal circuit, causal explanation | additive component-by-feature attribution; candidate for intervention | DLA composition is additive unless paired with interventions. |

## 3. Core Terms

### Core Method and Readout Objects

| Term | Category | Definition in this paper | Preferred usage | Avoid / Do not use | Literature grounding | First-use guidance |
|---|---|---|---|---|---|---|
| Sparse Readout Prism (SRP) | Paper-specific term | The paper's method: a sparse dictionary factorization of the final unembedding / LM head that decomposes selected linear readout scores into signed feature terms plus explicit residuals. | Use as the method name after spelling it out once. Emphasize "sparse feature decomposition for logit-lens readouts". | Do not describe SRP as a new language model, a new decoder-only lens, a general explanation method, or a causal circuit method. | New paper-specific term built from standard sparse dictionary, logit-lens, and output-embedding objects. | Introduce in Abstract and Introduction with: "We introduce Sparse Readout Prism (SRP), a sparse feature decomposition for logit-lens readouts..." Cite logit lens and SAE literature nearby. |
| logit lens | Standard literature term | A lens-style inspection method that maps hidden states through the model's unembedding / LM head to vocabulary logits or distributions. | Use when referring to the standard interface SRP extends. | Do not use "logit lens" to refer to SRP's sparse decomposition itself. | Standard term from `nostalgebraist2020`; related to tuned/future/patchscope lens work `belrose2023tuned,pal2023future,ghandeharioun2024patchscopes`. | Define in Introduction before SRP. Citation required. |
| lens-style readout | Standard literature term / needs clarification | A broader family of methods that inspect hidden states via vocabulary scores, distributions, continuations, or descriptions. | Use for the class of prior interfaces, especially in Related Work. | Do not let it blur with "readout" as the LM head. | Grounded in logit lens, tuned lens, Future Lens, Patchscopes, and newer lens variants. | First paragraph of Introduction or Related Work. Citation required. |
| readout | Needs clarification | The final unembedding or LM head map from hidden states to vocabulary logits. In this paper, "readout" is the final interface, not the whole model. | Use as the concise paper-wide alias for final unembedding / LM head. | Do not use for the entire model, hidden-state computation, generated text, or task behavior. | Standard-ish term, narrowed as a paper-scoped alias; grounded in output embedding and LM-head literature. | Define in the first Introduction paragraph: "we call this interface the readout." Cite output-embedding work in Method or Related Work. |
| unembedding matrix, `W_U` | Standard literature term | The matrix whose token rows map a hidden state to vocabulary logits. Row `t` is `w_t^T`; token `t`'s logit is `h^T w_t`. | Use in mathematical sections and when discussing the fitted object. | Do not alternate casually with "embedding matrix" unless discussing tied weights. | Standard transformer / output-embedding terminology. Cite `press2017outputembedding,inan2017tying,yang2018softmaxbottleneck,bostrom2020bpe` when discussing output geometry. | Method, before the factorization equation. Citation useful, not required for the bare definition. |
| LM head | Standard literature term | The final linear readout layer that maps the final hidden state to vocabulary logits. | Use interchangeably with "final unembedding" when discussing replacement fidelity or implementation. | Do not use for the full next-token distribution after softmax or softcap unless explicitly stated. | Standard LLM implementation term. | Define alongside "readout" in Introduction or Method. Citation usually not needed. |
| output embedding | Standard literature term | The learned output-side token vectors / classifier weights that produce logits, especially when discussing geometry and weight tying. | Use in literature positioning and caveats about vocabulary geometry. | Do not use as the main paper term for the object being factorized; use "unembedding" or "LM head". | Standard output-embedding literature. | Related Work. Citation required when invoking geometry or tying. |
| sparse dictionary factorization | Standard literature term / adapted | A sparse representation of unembedding rows as shared decoder directions with sparse token-row coefficients. | Use for the technical move underlying SRP. | Do not call it a "causal decomposition" or "semantic discovery" without caveats. | Dictionary learning and SAE literature `makhzani2013ksparse,bricken2023monosemanticity,cunningham2023sparse,gao2024scaling`. | Introduction and Method. Citation required. |
| sparse autoencoder (SAE) | Standard literature term | The sparse autoencoder machinery used to reconstruct unembedding rows. | Use for the model class; specify that this SAE is trained on unembedding rows. | Do not imply it is an activation SAE unless discussing the contrast. | Standard in modern mechanistic interpretability. | Introduction or Method. Citation required. |
| readout SAE | Paper-specific term | The SAE trained on rows of the unembedding / readout matrix, rather than on hidden-state activations. | Use for the trained factorizer object. | Do not call it an "activation SAE". | Paper-specific specialization of SAE machinery. | Method, after defining the row factorization. Citation to SAE literature nearby. |
| TopK SAE | Standard literature term | An SAE whose encoder retains the `k` largest code entries, giving a fixed active feature budget. | Use when discussing the trained recipe and native `k`. | Do not use "TopK" for a display cutoff in figures. | Grounded in k-sparse autoencoders and TopK SAE work `makhzani2013ksparse,gao2024scaling`. | Method and appendix selection discussion. Citation required. |
| native `k` | Needs clarification | The active-feature budget of the trained TopK SAE. | Use when distinguishing trained sparsity from plotted bar truncation. | Do not call figure truncation "native k". | Adapted from TopK SAE practice. | Experimental Setup or appendix selection overview. Citation not needed if TopK already cited. |
| dictionary width, `D` | Needs clarification | Number of readout SAE decoder directions/features. Width is often reported as a multiplier of model width, e.g. `32*d_model`. | Use when describing capacity and sweep settings. | Do not confuse dictionary width with active feature budget `k`. | Standard SAE/dictionary capacity term, with paper-specific reporting convention. | Experimental Setup and appendix selection overview. |
| strict-budget regime | Paper-specific term | Operating regime using native `k=128` for fixed budget comparability. | Use only when comparing dictionary operating points. | Do not use as a synonym for "better interpretability" without metrics. | Paper-specific experimental setting. | Experimental Setup or appendix selection overview. No citation needed. |
| high-fidelity regime | Paper-specific term | Operating regime using native `k=256` when readout fidelity is the primary constraint. | Use only with replacement / reconstruction metrics. | Do not imply it is universally preferable. | Paper-specific experimental setting. | Experimental Setup or appendix selection overview. No citation needed. |
| readout feature | Paper-specific term | A reusable SAE decoder direction learned from unembedding rows; used by token rows and projected against hidden states. | Use as the intuitive name for `d_i` after defining it. | Do not equate with hidden-state SAE feature or semantic feature. | Paper-specific specialization of SAE decoder direction. | Introduction and Method. Define before examples. Citation to SAE literature nearby. |
| SAE decoder direction, `d_i` | Standard literature term | A learned dictionary direction; in SRP it is a readout-side direction in hidden-state space. | Use in mathematical prose; pair with "readout feature direction" for readability. | Avoid "neuron" or "latent concept" unless separately justified. | Standard SAE / dictionary-learning terminology. | Method factorization. Citation required when SAE introduced. |
| sparse code entry / token-row coefficient, `z_{t,i}` | Standard literature term / adapted | Sparse coefficient with which token row `t` uses readout feature `i`. | Use "sparse code entry" in SAE contexts and "token-row coefficient" in readout-target contexts. | Do not call it a hidden-state activation. | Standard sparse coding term, adapted to unembedding rows. | Method factorization. Citation to SAE literature. |
| shared offset, `mu` | Standard technical term | Common row offset in the raw-space row reconstruction; cancels for zero-sum contrasts. | Use in score identities and when explaining raw token logits. | Do not hide it as a dense correction term. | Standard centering / affine reconstruction term. | Method factorization and score decomposition. Citation not needed. |
| raw readout coordinates | Needs clarification | The model's original logit/contrast coordinate system after mapping SAE quantities back from any centering or row normalization. | Use to reassure readers that displayed scores are in original model units. | Do not imply preprocessing is absent; say quantities are mapped back. | Paper-specific bookkeeping phrase. | Method after training preprocessing. No citation needed. |
| row reconstruction | Standard literature term / needs clarification | Reconstruction of held-out unembedding rows by the readout SAE. | Use as a capacity and fit metric, not as proof that local decompositions are valid. | Do not conflate with target reconstruction or LM-head replacement fidelity. | Standard reconstruction objective/evaluation term. | Experimental Setup. Citation to SAE literature optional. |
| row reconstruction objective | Standard literature term / adapted | The mean-squared reconstruction objective used to train the readout SAE on preprocessed unembedding rows. | Use when describing SAE training. | Do not present it as an objective for end-to-end language modeling or task behavior. | Standard autoencoder objective applied to readout rows. | Method factorization subsection. Cite SAE literature nearby. |
| rowEV | Needs clarification | Row-centered explained variance for reconstructed held-out unembedding rows. | Use as a row reconstruction capacity metric paired with replacement and target fidelity. | Do not use as the only evidence that feature bars are interpretable. | Adapted metric label. | Define in Results table caption and glossary. No citation needed. |

### Target and Score Decomposition Terms

| Term | Category | Definition in this paper | Preferred usage | Avoid / Do not use | Literature grounding | First-use guidance |
|---|---|---|---|---|---|---|
| readout feature projection, `p_i(h)=h^T d_i` | Paper-specific term | The projection of hidden state `h` onto readout feature direction `d_i`. | Use for hidden-state coordinates induced by the readout SAE. | Do not call it a feature activation in technical prose. | Paper-specific use of standard projection onto SAE decoder directions. | Introduction and Method. Define before target coefficients. |
| readout feature projection vector, `p(h)=D_feat h` | Paper-specific term | The vector of all readout feature projections before a scalar target is selected. | Use to distinguish target-independent hidden-state projections from target-specific signed contributions. | Avoid "activation vector" or "evidence vector". | Paper-specific but mathematically direct. | Method feature-projection subsection. No citation needed beyond SAE. |
| projection profile | Needs clarification | Informal shorthand for the ranked pattern of readout feature projections before target selection. | Use only after "readout feature projection vector" is defined. | Do not use as the formal mathematical term. Avoid "profile activation". | Paper-local shorthand. | If used, introduce in appendix/display text only. No citation needed. |
| scalar readout target | Paper-specific term | A chosen scalar linear function of unembedding rows, `q_alpha=sum_t alpha_t w_t`, scored as `s_alpha(h)=h^T q_alpha`. | Use as the formal term for the quantity decomposed on the fixed SRP basis. | Avoid "query" as the main formal term; avoid implying nonlinear quantities are directly supported. | Paper-specific formalization of standard linear logits and contrasts. | Method, before target families. No citation needed. |
| readout target | Paper-specific shorthand | Short form of "scalar readout target" after first definition. | Use when the scalar/linear nature is already clear. | Do not use for arbitrary benchmark tasks or labels without specifying the linear contrast. | Paper-specific shorthand. | Only after "scalar readout target" is defined. |
| token logit | Standard literature term | The scalar score `h^T w_A` for token `A`. | Use for raw token score targets. | Do not confuse with probability after softmax. | Standard language-model terminology. | Method target families. Citation not needed. |
| centered logit / reference contrast | Standard term / adapted | A token logit after subtracting a reference row, usually vocabulary mean or shared offset. | Prefer "reference contrast" when a particular reference row is defined. | Do not call it a probability-normalized score. | Standard linear contrast idea; paper-specific target family. | Method target families. Cite output-embedding caveats if motivating. |
| pairwise logit difference | Standard literature term | The contrast `h^T(w_A-w_B)`; positive values support `A` over `B`. | Use for token-token signed margins. | Avoid "pairwise preference" unless clarifying that it is a logit difference, not a human preference label. | Standard in logit attribution and lens analyses. | Introduction and Method. Citation optional. |
| group contrast | Standard term / adapted | A linear contrast between average rows for two token groups. | Use for answer-vs-distractor, abstention-vs-entity, label, or action-family readout targets. | Do not imply it measures end-to-end group task performance. | Adapted from linear contrast terminology. | Method target families and benchmark-derived target section. Citation for datasets, not for the term. |
| local competitor margin / top-competitor margin | Needs clarification | A contrast between a selected token and fixed nearby competitors from the original LM-head ranking. | Prefer one canonical phrase: "top-competitor margin" in tables and "local competitor margin" only when emphasizing local ranking. | Do not call it a classification margin unless the candidate set and weights are specified. | Adapted margin concept. | Method target families. No citation needed. |
| target coefficient, `beta_i(q)` | Paper-specific term | The coefficient with which scalar target `q` uses readout feature `i`; e.g. `sum_t alpha_t z_{t,i}`. | Use when discussing the readout/target-side factor. | Do not call it an activation or context effect. | Paper-specific term derived from sparse row coefficients. | Method local score decomposition. |
| signed feature contribution, `c_i(h,q)` | Paper-specific term | The target coefficient times the hidden-state projection, `beta_i(q) p_i(h)`, positive or negative for the chosen scalar target. | Use for bars in local score plots and analysis of support/opposition. | Avoid "feature importance" unless tied to the selected target and residual. | Paper-specific contribution term. | Introduction and Method. |
| support / opposition | Needs clarification | Directional contribution relative to a specified signed target: positive terms raise the chosen scalar; negative terms lower it. | Use only after the target and sign convention are stated. | Do not imply moral, behavioral, or causal support. | Standard sign interpretation adapted to SRP. | Figure captions and Method. |
| local score decomposition | Paper-specific term | The target-conditioned decomposition of one selected scalar score at one hidden state into offset, signed feature terms, and residual. | Use for displayed, target-local SRP decompositions. | Do not call it a global explanation of model behavior. | Paper-specific term needed to avoid overclaiming "explanation". | Introduction, at first example. No citation needed. |
| score decomposition at a hidden state | Paper-specific / standard mathematical phrase | The mathematical identity underlying a local score decomposition. | Use in formal sections and appendices. | Do not alternate with "explanation" unless scoped. | Paper-specific instantiation of linear decomposition. | Method local score decomposition. |
| feature sum, `s_feat` | Paper-specific bookkeeping term | Sum of all signed feature contributions before adding offset or residual terms. | Use in figure diagnostics and score decomposition. | Do not confuse with displayed top bars if bars are truncated. | Paper-specific reporting term. | Method residual-diagnostics subsection or appendix. |
| reconstruction residual / reconstruction error, `epsilon` | Standard literature term | Difference between the exact target score and reconstructed score from offset plus feature sum. | Prefer "reconstruction error" for scalar target error; "residual" for the explicit leftover term. | Do not use "unexplained behavior"; it is unmodeled readout score under the sparse factorization. | Standard reconstruction terminology. | Method fidelity diagnostics. Citation optional. |
| relative reconstruction error, `rho` | Needs clarification | Normalized scalar target reconstruction error, `abs(epsilon)/(abs(s_exact)+delta)`, with `delta=0.5` logit in aggregate summaries. | Use for aggregate and figure reliability reporting. | Do not compare values if denominator conventions differ. | Paper-specific metric convention. | Results before first table or table caption. |
| sign agreement / sign flip | Standard evaluation term | Whether reconstructed signed target has the same sign as the exact target; a flip reverses the supported side. | Use for signed contrasts and margins. | Do not use for unsigned raw logits without defining a threshold. | Standard binary agreement term. | Method residual diagnostics and Results. |

### Metrics, Evaluation, and Display Terms

| Term | Category | Definition in this paper | Preferred usage | Avoid / Do not use | Literature grounding | First-use guidance |
|---|---|---|---|---|---|---|
| LM-head replacement fidelity | Needs clarification | Evaluation of substituting reconstructed `W_U` for the original LM head on held-out hidden states and comparing readout behavior. | Use for global readout behavior preservation. | Do not use as a synonym for local target fidelity. | Paper-specific metric grouping over standard top-1/KL comparisons. | Experimental Setup and Results. |
| target reconstruction / target fidelity | Needs clarification | Exact-versus-reconstructed score behavior for held-out scalar readout targets. | Use as the primary evidence for interpreting local decompositions. | Do not conflate with task accuracy or generation quality. | Paper-specific metric grouping using standard reconstruction and sign metrics. | Experimental Setup and Results. |
| replacement metrics: top-1 agreement, top-5 overlap, KL | Standard evaluation terms | Metrics comparing original and reconstructed LM-head outputs on held-out hidden states. | Use with exact definitions and units; KL is in bits in appendices. | Do not call top-5 overlap "top-5 accuracy". | Standard ranking/divergence metrics. | Results table captions. Citation not needed. |
| null baseline | Standard evaluation term | A comparison designed to destroy task-relevant structure, such as shuffled sparse codes or random support. | Use only for methods expected to remove learned token-to-feature support. | Do not call PCA or nearest-row ridge "null" if they retain structured information. | Standard experimental terminology. | Results baseline section. |
| reference baseline | Standard evaluation term | A structured comparison such as nearest-row ridge or PCA that retains some output-embedding geometry. | Use for non-null comparison methods. | Do not use "baseline" for arbitrary alternatives not evaluated. | Standard evaluation terminology; PCA/ridge have their own literature. | Results baseline section. Cite `jolliffe2002principal,hoerl1970ridge`. |
| learned token-to-feature support | Needs clarification | The learned pattern of sparse code support connecting token rows to readout SAE features. | Use when explaining why shuffled/random controls matter. | Avoid "semantic support" unless supported by labels and audits. | Paper-specific phrasing over sparse code support. | Results baseline section. |
| benchmark-derived readout target | Paper-specific term | A scalar readout target constructed from benchmark examples, e.g. answer-vs-distractor or safe-vs-unsafe token groups. | Use to stress readout-level evaluation of benchmark-shaped contrasts. | Do not call these "benchmark performance" or "task accuracy". | Paper-specific experimental setting; datasets are standard. | Results section on SQuAD2, HotpotQA, LegalBench/ContractNLI, SecurityEval. Cite datasets. |
| target set | Needs clarification | Fixed evaluation input containing prompts, token targets/groups, linear coefficients, and tokenization filters. | Use in Experimental Setup and Reproducibility. | Do not use interchangeably with dataset if targets are constructed from datasets. | Paper-specific evaluation bookkeeping. | Experimental Setup. |
| evaluation item | Needs clarification | One evaluated readout target after filters; prompt variants and target variants may share a base case. | Use for row counts. | Do not call every prompt an evaluation item if multiple targets derive from it. | Paper-specific bookkeeping. | Appendix or tables. |
| base case | Needs clarification | Cluster of variants from the same prompt/target template used as bootstrap unit. | Use in confidence interval or aggregation discussions. | Do not use as "baseline". | Paper-specific evaluation bookkeeping. | Robustness appendix. |
| feature label | Needs clarification | A short readability label derived from vocabulary rows strongly associated with a readout SAE feature. | Use only as a display aid; keep feature id as the measured object. | Do not treat labels as ground-truth semantics or discovered concepts. | Adapted from sparse-feature labeling practice. | Method final paragraph, figure captions, limitations. |
| token audit item | Needs clarification | Tokenizer or row factor that can affect interpretation: token length, frequency, row norm, capitalization, whitespace, punctuation, special token status, or tokenization collision. | Use in robustness controls and limitations. | Do not leave brittle single-token claims unaudited. | Paper-specific control grouping grounded in tokenizer/output-embedding caveats. | Robustness appendix and limitations. |
| tokenization collision | Needs clarification | A case where tokenization makes an intended lexical contrast brittle or ambiguous. | Use when replacing a brittle single-token contrast with a group contrast. | Do not silently drop or reinterpret affected examples. | Standard tokenizer caveat, paper-specific term. | Robustness appendix. |

### Attribution, Edits, and Model-Specific Terms

| Term | Category | Definition in this paper | Preferred usage | Avoid / Do not use | Literature grounding | First-use guidance |
|---|---|---|---|---|---|---|
| direct logit attribution (DLA) | Standard literature term | Additive attribution that projects residual-stream components onto a selected readout direction. | Use for component attribution, not for SRP itself. | Do not imply causal effect unless paired with interventions. | Standard mechanistic interpretability term `elhage2021framework,wang2022ioi,nguyen2024logitprisms`. | Results section on feature-resolved attribution. Citation required. |
| feature-resolved DLA / DLA by SAE feature | Paper-specific term | SRP's decomposition of DLA component contributions across readout SAE features. | Prefer "feature-resolved DLA" in main prose; "DLA by SAE feature" is acceptable for appendix titles. | Do not call it circuit discovery or causal tracing. | Paper-specific composition of SRP with standard DLA. | Results section after DLA is defined. |
| additivity error | Standard / needs clarification | Gap between exact target score and the sum of component attribution terms under the DLA decomposition. | Use separately from readout SAE reconstruction error. | Do not merge it with SRP residual unless explicitly summing error sources. | Standard attribution bookkeeping. | DLA appendix or Results. |
| constrained readout-side edit | Needs clarification | A readout-level intervention along selected SAE decoder directions, evaluated on specified lexical scores. | Use for the stress-test experiment. | Avoid "behavioral control", "safety filter", or "generation control" unless separately evaluated. | Paper-specific experimental phrasing related to steering/intervention language. | Results readout-edit section and limitations. |
| readout-level intervention | Needs clarification | An intervention that changes specified readout scores, not necessarily generated behavior or upstream mechanisms. | Use in the edit appendix when discussing causal effect at the edited score. | Do not generalize to system-level behavior. | Standard intervention concept scoped to readout-level scores. | Lexical-control appendix. |
| softcap-correct | Needs clarification | Gemma evaluation that applies the model's monotone final-logit softcap before top-1/KL comparisons. | Use only for Gemma readout evaluations that follow the actual output path. | Do not use for rowEV, which is measured on linear pre-softcap rows. | Model-specific implementation detail. | Experimental Setup / selection appendix. |
| reasoning-distilled checkpoint / readout | Standard term / needs clarification | A model checkpoint produced by reasoning-oriented distillation; used as a model-family/readout comparison row. | Use to identify Qwen/Llama distill comparisons. | Do not claim SRP explains "reasoning" from these rows. | Grounded in DeepSeek-R1 / distillation model literature. | Experimental Setup or model-suite appendix. Cite model source. |

## 4. Paper-Specific Terms

### Sparse Readout Prism (SRP)

- **Definition:** SRP is the paper's target-independent method: a sparse factorization of the final unembedding / LM head that supplies readout features and token-row coefficients. Selected scalar readout targets then turn this basis into signed feature terms plus explicit residuals.
- **Why this term is needed:** The paper needs a name for the factorization and projection basis, while keeping the selected-target interface explicit as the target-conditioned layer.
- **Closest standard term(s):** logit lens, sparse dictionary factorization, sparse autoencoder, direct logit attribution.
- **How it differs from those terms:** SRP is not just a lens decoder, not a generic SAE, and not component attribution. It factorizes the output readout itself; selected scalar readout targets use that factorization to produce target-local signed terms.
- **Where to introduce it:** Abstract and first half of Introduction, after defining the readout/logit-lens interface.
- **Suggested first-use sentence:** "We introduce **Sparse Readout Prism (SRP)**, a sparse feature decomposition for logit-lens readouts that factorizes the final unembedding / LM head; selected scalar readout targets then use this basis to decompose linear readout scores into signed feature terms plus explicit residuals."
- **Acceptable shorthand after first use:** SRP; the SRP factorization; SRP local decomposition; SRP readout basis.
- **Risks / reviewer concerns:** The name may sound broader than the evidence if the prose says SRP explains model behavior generally. Keep the claim at readout-linear score decomposition unless additional experiments are cited.

### readout SAE

- **Definition:** The sparse autoencoder trained on unembedding rows to reconstruct the final readout matrix.
- **Why this term is needed:** It distinguishes SRP's fitted object from activation SAEs trained on hidden-state activations.
- **Closest standard term(s):** sparse autoencoder; activation SAE; dictionary learning.
- **How it differs from those terms:** The training examples are token rows of `W_U`, so the sparse codes are token-row coefficients and the decoder directions are readout directions.
- **Where to introduce it:** Method subsection "Sparse Factorization of the Readout".
- **Suggested first-use sentence:** "We call the SAE trained on unembedding rows the **readout SAE**; unlike an activation SAE, its sparse codes describe token rows rather than hidden-state activations."
- **Acceptable shorthand after first use:** readout SAE; row SAE only in notes, not main prose.
- **Risks / reviewer concerns:** Reviewers may assume ordinary activation SAEs. Repeat the fitted-object distinction in Related Work and Limitations.

### readout feature

- **Definition:** A reusable SAE decoder direction learned from unembedding rows, denoted `d_i`.
- **Why this term is needed:** "SAE decoder direction" is accurate but cumbersome in figures and prose; "readout feature" marks that these features live in readout geometry.
- **Closest standard term(s):** SAE feature; decoder direction; dictionary atom.
- **How it differs from those terms:** It is a decoder direction from an SAE trained on unembedding rows, not a hidden-state activation feature.
- **Where to introduce it:** Introduction, immediately after the SRP factorization is described; Method gives the formal definition.
- **Suggested first-use sentence:** "We refer to the learned decoder directions `d_i` as **readout features** because they are features of the final readout matrix rather than hidden-state SAE activations."
- **Acceptable shorthand after first use:** feature, if the local context is clearly SRP/readout; decoder direction in equations.
- **Risks / reviewer concerns:** "Feature" can sound semantic or monosemantic. Pair qualitative labels with residuals and feature ids; avoid claiming ground-truth semantics.

### readout feature projection

- **Definition:** The scalar projection `p_i(h)=h^T d_i` of a hidden state onto a readout feature direction.
- **Why this term is needed:** It prevents confusion between hidden-state projections and SAE activations/codes.
- **Closest standard term(s):** projection; activation; feature coordinate.
- **How it differs from those terms:** It is not the readout SAE encoder activation. The SAE was trained on rows, so its activations are token-row codes `z_{t,i}`.
- **Where to introduce it:** Method subsection "Readout Reconstruction and Feature Projections".
- **Suggested first-use sentence:** "For a hidden state `h`, we call `p_i(h)=h^T d_i` the **readout feature projection**: it is the coordinate of `h` along readout feature `i` before any scalar target has been selected."
- **Acceptable shorthand after first use:** projection; projection coordinate; `p_i(h)`.
- **Risks / reviewer concerns:** If the draft says "activation", reviewers familiar with SAEs may object. Search and replace technical uses.

### readout feature projection vector

- **Definition:** The vector `p(h)=D_feat h` collecting all readout feature projections for a hidden state before target selection.
- **Why this term is needed:** It names the target-independent hidden-state view produced by the SRP basis.
- **Closest standard term(s):** feature vector; representation; activation vector.
- **How it differs from those terms:** It is induced by the readout SAE decoder directions and becomes evidence only after a target coefficient is supplied.
- **Where to introduce it:** Method subsection "Readout Reconstruction and Feature Projections".
- **Suggested first-use sentence:** "The vector `p(h)=D_feat h` is the **readout feature projection vector**, a target-independent set of projections that becomes token- or contrast-specific only after choosing a scalar readout target."
- **Acceptable shorthand after first use:** projection vector; projection profile for ranked/display contexts only.
- **Risks / reviewer concerns:** Avoid implying all large projections matter for the selected score.

### scalar readout target

- **Definition:** A scalar linear function of unembedding rows, `q_alpha=sum_t alpha_t w_t`, scored at hidden state `h` as `s_alpha(h)=h^T q_alpha`.
- **Why this term is needed:** The selected-target interface supports token logits, reference contrasts, pairwise differences, group contrasts, and top-competitor margins through one linear target formalism.
- **Closest standard term(s):** logit, logit difference, linear probe direction, readout direction, query.
- **How it differs from those terms:** It is restricted to linear functions of readout rows and is the exact unit decomposed by SRP.
- **Where to introduce it:** Method subsection "Scalar Readout Targets"; mention in Introduction after SRP.
- **Suggested first-use sentence:** "We call any selected linear function of unembedding rows a **scalar readout target**, `q_alpha=sum_t alpha_t w_t`, and score it at hidden state `h` as `s_alpha(h)=h^T q_alpha`."
- **Acceptable shorthand after first use:** readout target; target; selected scalar.
- **Risks / reviewer concerns:** Do not use it for probabilities, losses, task accuracy, or full generations unless a linear proxy is explicitly defined.

### target coefficient

- **Definition:** `beta_i(q)=sum_t alpha_t z_{t,i}`, the coefficient with which scalar target `q` uses readout feature `i`.
- **Why this term is needed:** It names the readout/target-side factor in the product `beta_i(q) p_i(h)`.
- **Closest standard term(s):** coefficient; sparse coefficient; loading.
- **How it differs from those terms:** It is not a single token-row sparse coefficient; it aggregates the target's row coefficients through `alpha`.
- **Where to introduce it:** Method subsection "Local Score Decomposition".
- **Suggested first-use sentence:** "For each feature `i`, the **target coefficient** `beta_i(q)=sum_t alpha_t z_{t,i}` records how the selected scalar target uses that readout feature."
- **Acceptable shorthand after first use:** coefficient, when unambiguous; `beta_i(q)`.
- **Risks / reviewer concerns:** It may be mistaken for learned classifier weights. State that it is derived from the fixed target and readout SAE after training.

### signed feature contribution

- **Definition:** `c_i(h,q)=beta_i(q) p_i(h)`, the signed local term contributed by readout feature `i` to target `q` at hidden state `h`.
- **Why this term is needed:** It is the central plotted quantity and the bridge between target-side coefficients and context-side projections.
- **Closest standard term(s):** attribution term; contribution; feature importance.
- **How it differs from those terms:** It is a readout-linear decomposition term, not a causal attribution unless paired with interventions.
- **Where to introduce it:** Method subsection "Local Score Decomposition"; also in Introduction around the main figure.
- **Suggested first-use sentence:** "The **signed feature contribution** is `c_i(h,q)=beta_i(q)p_i(h)`: positive terms raise the chosen scalar target and negative terms lower it."
- **Acceptable shorthand after first use:** signed contribution; feature term; contribution bar.
- **Risks / reviewer concerns:** Avoid "importance" without residual and target context; the contribution is local to one target and hidden state.

### local score decomposition

- **Definition:** A target-local decomposition of one selected readout score at one hidden state into offset, signed feature terms, and residual/error terms.
- **Why this term is needed:** It avoids overclaiming "explanation" while giving a stable name to the displayed SRP output.
- **Closest standard term(s):** local explanation; attribution; decomposition; decomposition identity.
- **How it differs from those terms:** It is an additive readout score decomposition, not a global behavioral explanation or causal mechanism.
- **Where to introduce it:** Introduction, with the first `verify` versus `assume` example.
- **Suggested first-use sentence:** "We call the signed, target-specific decomposition a **local score decomposition**: local because it decomposes one selected linear readout score at one hidden state and target."
- **Acceptable shorthand after first use:** local decomposition; score decomposition; SRP decomposition.
- **Risks / reviewer concerns:** "Decomposition" can still read as explanatory if residuals are hidden. Always pair it with reconstruction residual and claim scope.

### benchmark-derived readout target

- **Definition:** A scalar readout target built from benchmark examples, such as answer-vs-distractor, abstention-vs-entity, label, or safe-vs-unsafe token-group contrasts.
- **Why this term is needed:** It distinguishes readout-level reconstruction of benchmark-shaped targets from end-to-end benchmark performance.
- **Closest standard term(s):** benchmark task; evaluation target; task contrast.
- **How it differs from those terms:** It evaluates reconstruction of a constructed linear readout contrast, not whether the model solves the task.
- **Where to introduce it:** Results subsection "Scalar Readout Targets Extend Beyond Token Pairs".
- **Suggested first-use sentence:** "We use **benchmark-derived readout targets** for task-shaped linear contrasts derived from benchmark examples; these evaluate readout-target reconstruction, not end-to-end task accuracy."
- **Acceptable shorthand after first use:** benchmark-derived target; task-shaped readout contrast.
- **Risks / reviewer concerns:** Reviewers may object if prose implies benchmark success. Keep "readout-level" explicit.

### feature-resolved DLA

- **Definition:** The decomposition of direct logit attribution terms across SRP readout features for a realized forward pass.
- **Why this term is needed:** It names the composition of target-conditioned readout terms with residual-stream component attribution.
- **Closest standard term(s):** direct logit attribution; logit prism; component attribution.
- **How it differs from those terms:** DLA decomposes by residual component; feature-resolved DLA additionally splits each component contribution across readout SAE features.
- **Where to introduce it:** Results subsection "Readout Feature Terms Compose with Direct Logit Attribution".
- **Suggested first-use sentence:** "Because target-conditioned readout terms are linear in the final hidden state, we can form **feature-resolved DLA** by splitting residual-stream direct logit attribution across readout SAE features."
- **Acceptable shorthand after first use:** DLA by SAE feature; component-by-feature attribution.
- **Risks / reviewer concerns:** Do not present it as circuit discovery or causal abstraction. Say interventions are needed to test necessity.

### constrained readout-side edit

- **Definition:** A readout-level edit along selected SAE decoder directions, evaluated on specified lexical logit differences and controls.
- **Why this term is needed:** It describes the edit experiment without overclaiming behavioral control.
- **Closest standard term(s):** intervention; steering; activation steering; logit bias.
- **How it differs from those terms:** The edit is applied at the readout/score level using readout SAE directions and is evaluated on constrained lexical candidate scores.
- **Where to introduce it:** Results subsection "Selected Readout Features Move Held-Out Lexical Scores".
- **Suggested first-use sentence:** "We run a **constrained readout-side edit** that shifts logits along selected readout SAE decoder directions and evaluates held-out lexical logit differences."
- **Acceptable shorthand after first use:** readout edit; edit; stress test.
- **Risks / reviewer concerns:** "Steering" or "control" can imply generation-level reliability. Keep candidate-constrained scoring and off-target caveats visible.

## 5. Standard Terms to Use

| Preferred term | Short definition | Common alternatives | When alternatives are acceptable | Citation needed at first use? |
|---|---|---|---|---|
| logit lens | Decoding hidden states through the unembedding / LM head into vocabulary logits or distributions. | lens, lens-style decoding, vocabulary readout | "Lens-style readout" is acceptable for the broader family. | Yes: `nostalgebraist2020`; related variants `belrose2023tuned,pal2023future,ghandeharioun2024patchscopes`. |
| tuned lens | Learned affine/layer-specific lens for eliciting predictions from hidden states. | learned lens | Use only when discussing Belrose et al. or learned lens variants. | Yes: `belrose2023tuned`. |
| unembedding matrix | Matrix of output-side token rows used to compute logits. | LM head, output embedding | "LM head" is fine in implementation/evaluation contexts; "output embedding" in literature positioning. | Usually cite output-embedding work when discussing geometry. |
| LM head | Final map from hidden states to vocabulary logits. | final unembedding, readout | Use "readout" after definition. | No for implementation; yes when making output-geometry claims. |
| output embedding | Output-side token representation / classifier weights. | unembedding, output classifier | Use mostly in Related Work and caveats. | Yes: `press2017outputembedding,inan2017tying,yang2018softmaxbottleneck,bostrom2020bpe`. |
| sparse autoencoder | Autoencoder with sparse latent/code representation. | SAE, sparse dictionary | SAE after first use. | Yes: `makhzani2013ksparse,bricken2023monosemanticity,cunningham2023sparse,gao2024scaling`. |
| dictionary learning | Learning a set of basis/dictionary directions with sparse codes. | sparse dictionary factorization | Use when connecting to broader methods. | Yes if presented as literature. |
| row reconstruction objective | Mean-squared reconstruction objective for the readout SAE's preprocessed unembedding-row inputs. | reconstruction loss; autoencoder objective | Use when describing training, not interpretation. | Cite SAE literature at first SAE use. |
| decoder direction | Learned SAE dictionary direction. | SAE feature, dictionary atom | "Readout feature direction" is acceptable for SRP after definition. | Cite SAE literature at first SAE use. |
| sparse code | Sparse vector of coefficients produced by the SAE encoder. | code, coefficients, activations | In SRP, specify "token-row sparse code" because the SAE input is an unembedding row. | Cite SAE literature at first SAE use. |
| reconstruction error | Difference between original and reconstructed object or score. | residual, error | Use "residual" for the explicit leftover term in the identity. | No if standard; cite SAE literature if discussing evaluation practice. |
| explained variance | Fraction of variance captured by a reconstruction. | EV, rowEV | Use "rowEV" only for the paper's row-centered unembedding-row metric. | No. |
| top-1 agreement | Argmax agreement between two readout distributions/scores. | argmax agreement | Use with original vs reconstructed LM head. | No. |
| top-5 overlap | Mean overlap between top-five sets. | top-k overlap | Do not call it top-5 accuracy. | No. |
| KL divergence | Divergence between original and reconstructed readout distributions. | KL | State direction and units when relevant. | No for standard metric. |
| baseline | Evaluated comparison method or condition. | comparator, reference | Use only for methods in experiments. | Cite PCA/ridge if used. |
| null baseline | A control expected to remove learned structure. | null control | Use for shuffled sparse codes and random support. | No. |
| ablation | Removing or modifying a component to test its effect. | component removal | Use only when an SRP component or training choice is altered; not for unrelated baselines. | No. |
| direct logit attribution | Residual-stream component projection onto a target readout direction. | DLA, component attribution | DLA after first use. | Yes: `elhage2021framework,wang2022ioi,nguyen2024logitprisms`. |
| causal intervention | Deliberate intervention used to test necessity/effect. | causal test, intervention | Use only for actual edits/interventions, not additive decompositions. | Yes if connecting to causal abstraction/circuit tracing: `geiger2023causalabstraction,ameisen2025circuittracing`. |
| activation SAE | SAE trained on hidden-state activations. | hidden-state SAE | Use when contrasting with readout SAE. | Yes: `bricken2023monosemanticity,cunningham2023sparse,gao2024scaling`. |
| BatchTopK SAE | SAE variant used as a recipe comparison/control. | BatchTopK | Use only when reporting rejected or comparison recipes. | Yes: `bussmann2024batchtopk`. |
| JumpReLU SAE | SAE variant with JumpReLU gating used as a recipe comparison/control. | JumpReLU | Use only when reporting recipe sweeps. | Yes: `rajamanoharan2024jumprelu`. |
| Matryoshka SAE | Multi-level SAE recipe used as an architectural reference/comparison. | Matryoshka TopK | Use only when discussing sweep comparisons. | Yes: `bussmann2025matryoshka`. |
| orthogonal matching pursuit | Sparse-inference comparison over fixed dictionaries. | OMP | Use as a comparison method, not as the deployed SRP encoder. | Yes: `pati1993orthogonal`. |
| PCA | Dense linear subspace baseline/reference. | principal components | Use as a reference baseline, not a null baseline. | Yes: `jolliffe2002principal`. |
| ridge regression | Regularized least-squares reference model. | nearest-row ridge | Use for the nearest-row ridge reference. | Yes: `hoerl1970ridge`. |
| C4 continuations | Held-out hidden states from C4-derived text continuations used for replacement evaluation. | neutral C4 continuations | Use only for the replacement evaluation data. | Yes: `raffel2020exploring,dodge2021documenting`. |
| SQuAD2 | QA dataset with answerable and unanswerable questions. | SQuAD 2.0 | Use only for dataset-derived readout targets. | Yes: `rajpurkar2018squad2`. |
| HotpotQA | Multi-hop QA dataset. | Hotpot | Use formal name in paper. | Yes: `yang2018hotpotqa`. |
| ContractNLI | Contract natural-language inference dataset. | legal NLI | Use with LegalBench when applicable. | Yes: `koreeda2021contractnli`. |
| LegalBench | Legal reasoning benchmark collection. | legal benchmark | Use for the benchmark source, not as proof of legal reasoning. | Yes: `guha2023legalbench`. |
| SecurityEval | Code-generation security benchmark/dataset. | security dataset | Use for safe/unsafe action readout targets only. | Yes: `siddiq2022securityeval`. |

## 6. Discouraged or Risky Terms

| Term | Problem | Why reviewers may object | Safer alternative | When it is acceptable |
|---|---|---|---|---|
| feature activation | Misleading for `h^T d_i`; in SAE literature, activation usually means encoder output. | The readout SAE is trained on unembedding rows, so hidden-state projections are not SAE activations. | readout feature projection; projection coordinate | Acceptable only for token-row SAE code entries if clearly referring to encoder outputs on unembedding rows. |
| profile activation | Nonstandard and doubly confusing. | It mixes "projection profile" with SAE activation language. | projection profile; readout feature projection vector | Avoid in main text. |
| explanation | Too broad if used without scope. | The fixed SRP basis plus a selected target decomposes selected linear readout scores, not arbitrary model behavior. | local score decomposition; score decomposition; readout-level decomposition | Acceptable when scoped: "local explanation for a selected linear readout score" and accompanied by residuals. |
| explains model behavior | Overclaims beyond evidence. | The method does not establish sequence-level behavior, task accuracy, or mechanisms by itself. | decomposes selected readout scores; decomposes selected logits/contrasts | Only with strong qualifiers and separate behavioral evidence. |
| causal explanation | Unsupported by additive decomposition alone. | Causal claims require interventions or an identification argument. | additive decomposition; candidate mechanism for later intervention | Acceptable only for the readout-edit score intervention or separately tested causal interventions. |
| causal circuit | Stronger than feature-resolved DLA. | DLA by SAE feature is additive attribution over a forward pass, not a necessity/sufficiency test. | component-by-feature attribution; candidates for intervention | Acceptable only after causal interventions establish circuit claims. |
| faithful | Evidence-sensitive. | Reviewers may ask faithful to what, under what metric, and at what scale. | preserves the selected readout score with small residual; sign-preserved low-error decomposition | Acceptable when paired with reconstruction error, sign agreement, replacement fidelity, and target scope. |
| faithfulness | Same issue as "faithful"; can imply global interpretability. | Local residuals do not guarantee global model explanation. | reconstruction fidelity; target fidelity; replacement fidelity | Acceptable for named diagnostics after definition. |
| robust | Requires perturbation/distribution/seed evidence. | The paper has controls and some cross-family/seed evidence, but not arbitrary robustness. | consistent across tested settings; controlled by tokenization/frequency checks | Acceptable in "Fidelity Checks and Controls" as appendix/control framing; qualify each claim. |
| general | Vague and broad. | The method is evaluated on several target families and model families, not all tasks/models. | across evaluated target families; beyond token pairs; in the tested model suite | Acceptable only with the scope stated. |
| scalable | Requires explicit scaling evidence in size/data/compute/tasks. | Sweeps vary width, `k`, model family, and model size, but the paper should not imply unlimited scaling. | tested across dictionary widths/model families; higher-capacity setting | Acceptable if referring to a measured axis and table. |
| efficient | Requires cost/time/data comparison. | The paper reports fidelity metrics, not a primary efficiency claim. | compact; fixed-budget; lower dictionary width; lower active-feature budget | Acceptable when supported by compute/cost numbers or budget comparisons. |
| optimal / best | Too absolute. | Sweeps are finite and local to recipes/settings. | selected; strongest among evaluated settings; best in our sweep | Acceptable with explicit search space and metric. |
| state-of-the-art | Promotional and unsupported. | The paper does not benchmark against all interpretability methods. | competitive with baselines on our target reconstruction metrics | Avoid unless a full benchmark justifies it. |
| novel | Usually unnecessary and can irritate reviewers. | The contribution should be shown by definitions and comparisons. | new in this paper; introduced here; paper-specific | Use sparingly in contribution statement if venue style permits. |
| first | Strong priority claim. | Requires exhaustive literature search. | to our knowledge; we introduce | Use only if verified and scoped narrowly. |
| framework | Overused and vague. | SRP is mainly a method/factorization/decomposition identity, not necessarily a broad framework. | method; factorization; decomposition identity; formalism | Acceptable for Patchscopes title or if describing the scalar-target formalism broadly. |
| pipeline | Implementation-flavored and vague. | It obscures the mathematical contribution. | method; evaluation sequence; procedure | Acceptable only for reproducibility/implementation descriptions. |
| system | Implies deployed software or full model. | SRP is not a standalone system in the paper. | method; readout factorization; analysis | Avoid unless discussing software. |
| agent | Not applicable to final readout analysis. | The model is not acting autonomously in the experiments. | model; language model; checkpoint | Avoid. |
| generator | Can blur policy/generation behavior with readout scores. | The paper evaluates readout scores, not generators. | language model; model; LM head | Use only when discussing open-generation probes. |
| aligned | Overloaded in alignment literature. | SRP is not optimizing preferences or alignment. | matches selected readout target; sign-preserved; consistent with intended contrast | Avoid unless discussing "tokenizer-aligned" or unrelated technical alignment. |
| reasoning | Can imply cognitive reasoning. | Reasoning-distilled checkpoints are model sources, not a claim that SRP explains reasoning. | reasoning-distilled checkpoint; post-training comparison | Acceptable as part of model names or cited model descriptions. |
| semantic feature | Too strong for row-label summaries. | Feature labels are token-row summaries, not ground-truth semantics. | feature label; row-level descriptor; vocabulary-row summary | Acceptable only with caveats and audits. |
| interpretable feature | Evidence-sensitive. | Interpretability depends on labels, audits, residuals, and scope. | labeled readout feature; readable feature label | Acceptable when label status and reconstruction quality are clear. |
| control | Can imply deployment/safety control. | The readout edit is constrained and score-level. | constrained readout-side edit; readout-level intervention; lexical score movement | Acceptable as "control" only with "readout-side" and caveats. |
| steering | Often implies broad behavioral steering. | The edit experiment mostly measures candidate-constrained lexical scores. | readout edit; readout-level intervention | Acceptable in appendix title/prose if paired with "constrained" and score-level scope. |
| benchmark performance | Not what benchmark-derived targets evaluate. | The table reconstructs readout contrasts, not task accuracy. | benchmark-derived readout-target fidelity; readout-level contrast reconstruction | Avoid unless actual task performance is measured. |
| task accuracy | Unsupported for benchmark-derived readout targets. | The paper does not evaluate end-to-end answers in those rows. | sign reconstruction; target fidelity; contrast reconstruction | Use only if true end-to-end task accuracy is added. |
| preference | Overloaded with human preference/RLHF. | Pairwise logit differences are model readout contrasts, not preference labels. | logit difference; token preference if explicitly a readout preference | Use "preference" only informally after defining it as a logit-lens preference. |
| query | Overloaded with prompts/database queries; weaker than target. | The formal object is a scalar linear target. | scalar readout target | Acceptable in informal notes or if "query" is explicitly defined as the target. |
| margin | Needs specification. | Different margins mean pairwise, group, or top-competitor comparisons. | pairwise logit difference; top-competitor margin; group contrast | Acceptable when the compared sides and weights are specified. |

## 7. Synonym and Consistency Map

| Concept | Preferred term | Allowed variants | Disallowed variants | Notes |
|---|---|---|---|---|
| Full method | Sparse Readout Prism (SRP) | SRP; SRP factorization; SRP local decomposition | prism alone before definition; sparse prism if undefined | Spell out once in Abstract/Introduction. |
| Final output map | readout | final unembedding; LM head; unembedding matrix when mathematical | model; output layer without definition; decoder | Define readout as final unembedding / LM head. |
| Matrix being factorized | unembedding matrix, `W_U` | LM head matrix; output embedding matrix in literature context | embedding matrix if input embeddings are meant | Use `W_U` in equations. |
| Learned row factorizer | readout SAE | SAE trained on unembedding rows; row-factorization SAE | activation SAE; hidden-state SAE | Repeat contrast with activation SAEs. |
| Dictionary direction | readout feature | SAE decoder direction; readout feature direction; decoder direction `d_i` | semantic feature; neuron; concept | "Feature" is shorthand only after defining row-fitted object. |
| Token-row coefficient | sparse code entry | token-row coefficient; sparse coefficient; `z_{t,i}` | hidden-state activation; projection | Codes belong to unembedding rows. |
| Hidden-state coordinate | readout feature projection | projection coordinate; `p_i(h)=h^T d_i` | feature activation; activation; code | This is a dot product with a learned decoder direction. |
| Projection vector | readout feature projection vector | projection vector; projection profile for ranked displays | activation vector; evidence vector | Not evidence until target coefficients are applied. |
| Quantity to decompose | scalar readout target | readout target; selected scalar; target `q` | query as formal term; task; label | Must be linear in readout rows unless proxy is stated. |
| Target-side feature weight | target coefficient | coefficient; `beta_i(q)` | activation; context coefficient | Fixed by target and readout SAE. |
| Local plotted term | signed feature contribution | signed contribution; feature term; contribution bar | importance score; causal effect | Local to one hidden state and target. |
| Displayed decomposition | local score decomposition | local decomposition; score decomposition; readout decomposition | global explanation; causal explanation | Always report residual/error. |
| Positive/negative bars | support/opposition | supports first side; supports comparison side | good/bad evidence; causal support | Sign depends on target convention. |
| Error in SRP score decomposition | reconstruction error | residual; readout SAE residual; `epsilon` | noise; unexplained behavior | Keep separate from DLA additivity error. |
| Normalized local error | relative reconstruction error, `rho` | relative error | accuracy; loss | State denominator convention. |
| Signed-target correctness | sign agreement | sign preservation; sign flip rate | accuracy without context | Use only for signed contrasts/margins. |
| Global LM-head substitution metric | LM-head replacement fidelity | readout replacement; replacement metrics | target fidelity; row reconstruction | Uses held-out hidden states. |
| Local target metric | target reconstruction | target fidelity; scalar-target fidelity | replacement fidelity; task performance | Exact vs reconstructed scalar target. |
| Output distribution comparison | KL | KL divergence; KL in bits | loss unless specified | State direction if necessary. |
| Row reconstruction metric | rowEV | row-centered explained variance | fidelity by itself | RowEV is capacity/fit, not sufficient for local interpretation. |
| Comparison methods | baselines | null baselines; reference baselines | ablations unless a component is removed | Separate null vs reference baselines. |
| Dataset-derived contrast | benchmark-derived readout target | task-shaped readout contrast; benchmark-shaped target | benchmark result; benchmark accuracy | Readout-level only. |
| Feature label | feature label | row-level descriptor; token summary label | semantic label; ground-truth feature name | Feature id is stable object. |
| Attribution composition | feature-resolved DLA | DLA by SAE feature; component-by-feature attribution | causal circuit; circuit discovery | Additive unless interventions are performed. |
| Readout edit experiment | constrained readout-side edit | readout edit; readout-level intervention; lexical score movement | behavioral control; safety filter; policy steering | Keep candidate-constrained score scope. |
| Model collection | model suite | evaluated model suite; model-family comparison | benchmark suite unless benchmarked | Includes Qwen, Gemma, Ministral, reasoning-distilled rows. |
| Gemma output handling | softcap-correct | final-logit softcap-correct | calibration correction without definition | Applies to top-1/KL evaluation, not rowEV. |

## 8. Claim-Sensitive Terms

| Term | Evidence required | Current evidence in paper | Recommendation |
|---|---|---|---|
| faithful | Explicit target of faithfulness, reconstruction error, sign agreement for signed targets, and replacement fidelity where relevant. | Local decompositions report exact score, feature sum, residual, `rho`, and sign; replacement and target-fidelity tables exist. | Use as "faithful to the selected readout score" or "faithful enough to read under reported residuals"; avoid global "faithful explanation". |
| interpretable | Evidence that humans can read labels plus fidelity diagnostics and caveats. | Feature labels are row-level summaries; local decompositions have residuals; limitations state label caveats. | Prefer "readable under reported residuals", "residual-checked", or "diagnosed"; use "interpretability method" for field positioning only. |
| auditable | Explicit residuals, sign checks, and controls. | Strong: residuals, target reconstruction, sign agreement, token audits, null/reference baselines. | Use sparingly; prefer explicit residuals, sign checks, and audits. |
| robust | Perturbations, distributions, seeds, model families, or controls. | Controls include tokenization/frequency/row norm; some model-family and seed-window evidence. | Prefer "controlled" or "consistent across tested settings"; use "robustness controls" as appendix framing, not broad robustness claim. |
| general | Broad variation across tasks/models/settings. | Several target families and model families, but qualitative focus on Qwen and readout-level targets. | Use "beyond token pairs", "across evaluated target families", or "in the tested model suite". |
| scalable | Evidence across scale, data, compute, or task variation. | Sweeps vary model size, width, `k`, and some families. | Say "tested across dictionary widths and model families"; avoid "scalable" as a main claim. |
| efficient | Cost/time/data/compute comparison. | Fixed-budget/high-fidelity regimes, but not an efficiency benchmark. | Use "compact", "fixed-budget", or "native `k=128`" instead. |
| causal | Interventions or identification argument. | Additive DLA is non-causal; readout edit is an intervention on scores. | Use "causal" only for the constrained readout edit's score-level effect or when discussing future interventions. |
| controls / control | Defined intervention and measured off-target effects. | Readout edit moves constrained lexical scores; off-target probes and distribution shift are reported. | Say "constrained readout-side edit" or "readout-level intervention"; avoid deployment-control language. |
| explains | Clear object being explained and diagnostics. | The fixed SRP basis plus a selected target decomposes selected linear readout scores with residuals. | Prefer "decomposes" or "identifies feature terms for"; if using "explains", add "one selected readout score". |
| semantic | Human-interpretable labels with validation. | Feature labels derived from associated token rows and audited qualitatively. | Use "row-level descriptor" or "feature label"; avoid "semantic feature" unless qualified. |
| aligned | Preference/alignment evidence. | Not an alignment paper. | Avoid, except in technical phrases like "tokenizer-aligned". |
| reasoning | Behavioral reasoning evidence or model-name context. | The paper includes reasoning-distilled checkpoints only as comparison rows. | Use only in "reasoning-distilled checkpoint/readout"; do not claim SRP explains reasoning. |
| benchmark | Standard dataset/task evaluation. | Benchmark-derived readout targets are constructed from datasets, not full task performance. | Use "benchmark-derived readout target" and "readout-level fidelity"; avoid "benchmark performance". |
| state-of-the-art | Comprehensive benchmark against current methods. | Not present. | Avoid. |
| first | Exhaustive priority search. | Not established. | Avoid or use "to our knowledge" only for narrow, verified claims. |

## 9. First-Use Definitions

> We call the model's final unembedding or LM head the **readout**: the map from a hidden state to vocabulary logits.

> Following logit-lens work, we use **lens-style readouts** to refer to methods that inspect hidden states through vocabulary logits, distributions, continuations, or descriptions.

> We introduce **Sparse Readout Prism (SRP)**, a sparse feature decomposition for logit-lens readouts that factorizes the final unembedding / LM head and decomposes selected linear readout scores into signed feature terms plus explicit residuals.

> The **readout SAE** is a sparse autoencoder trained on unembedding rows, so its sparse codes describe token rows rather than hidden-state activations.

> We refer to the learned decoder directions `d_i` as **readout features** because they are features of the final readout matrix rather than activation-SAE features of hidden states.

> For a hidden state `h`, the **readout feature projection** `p_i(h)=h^T d_i` is its coordinate along readout feature `i` before any token, contrast, or margin has been selected.

> The vector `p(h)=D_feat h` is the **readout feature projection vector**, a target-independent set of projections that becomes evidence only after a scalar readout target supplies feature coefficients.

> We call any selected linear function of unembedding rows a **scalar readout target**, `q_alpha=sum_t alpha_t w_t`, and score it at hidden state `h` as `s_alpha(h)=h^T q_alpha`.

> A **pairwise logit difference** is the signed contrast `h^T(w_A-w_B)`, where positive values support token `A` over token `B`.

> A **group contrast** is a linear contrast between average unembedding rows for two token groups, used when the intended readout target is spread across several token rows.

> For each feature `i`, the **target coefficient** `beta_i(q)=sum_t alpha_t z_{t,i}` records how the selected scalar target uses that readout feature.

> The **signed feature contribution** is `c_i(h,q)=beta_i(q)p_i(h)`: positive terms raise the chosen scalar target and negative terms lower it.

> We call the signed, target-specific decomposition a **local score decomposition** because it decomposes one selected linear readout score at one hidden state and target.

> The **reconstruction error** `epsilon=s_exact-s_recon` is the part of the selected scalar target not captured by the offset and sparse feature terms.

> For signed contrasts, **sign agreement** means that the reconstructed scalar target has the same sign as the exact target, so the reconstructed decomposition supports the same side of the contrast.

> **LM-head replacement fidelity** measures whether substituting the reconstructed readout for the original LM head preserves held-out readout behavior, such as top-token agreement and KL.

> **Target reconstruction** measures exact-versus-reconstructed scores for held-out scalar readout targets and is the primary evidence for interpreting local score decompositions.

> A **feature label** is a row-level summary derived from vocabulary rows strongly associated with an SAE feature; the measured object remains the feature id and its signed contribution.

> We use **feature-resolved DLA** for the additive decomposition that splits residual-stream direct logit attribution terms across SRP readout features.

> A **benchmark-derived readout target** is a task-shaped linear contrast derived from benchmark examples; it evaluates readout-target reconstruction rather than end-to-end task accuracy.

> A **constrained readout-side edit** changes specified logits along selected readout SAE decoder directions and is evaluated as a readout-level intervention on lexical scores.

## 10. Terms to Check Against the Draft

- Search for "feature activation", "activations", and "profile activation". Replace technical uses of hidden-state `h^T d_i` with "readout feature projection" or "projection coordinate".
- Search for "query". Replace formal uses with "scalar readout target" unless "query" is clearly informal.
- Search for "explain", "explanation", and "explains". Ensure each use says what is explained: one selected linear readout score, not model behavior broadly.
- Search for "faithful" and "faithfulness". Verify each use specifies fidelity to the selected readout score, LM-head replacement behavior, or target reconstruction.
- Search for "robust", "robustness", and "general". Qualify with the tested controls, target families, model families, or operating regimes.
- Search for "causal", "cause", "circuit", and "mechanism". Keep DLA language additive unless an intervention is actually reported.
- Search for "control" and "steering". Replace broad uses with "constrained readout-side edit" or "readout-level intervention" unless generation-level behavior is evaluated.
- Search for "benchmark", "task accuracy", and "performance". For SQuAD2/HotpotQA/LegalBench/SecurityEval rows, say "benchmark-derived readout-target fidelity" unless measuring end-to-end task accuracy.
- Search for "preference". Use "pairwise logit difference" or "token preference under the readout" to avoid confusion with human preference labels.
- Search for "baseline" and "ablation". Use "baseline" only for evaluated comparison methods and "ablation" only when a component or recipe choice is removed/modified.
- Search for "framework", "pipeline", "system", and "approach". Prefer "method", "factorization", "formalism", "evaluation sequence", or "decomposition identity" where more precise.
- Search for "feature label", "semantic", and label descriptions. Confirm labels are described as row-level summaries, not ground-truth semantics.
- Search for "rowEV". Ensure it is never the sole justification for interpreting local feature bars.
- Search for "top-5". Ensure top-5 overlap is not called top-5 accuracy.
- Search for undefined acronyms: SRP, SAE, DLA, KL, PCA, OMP, C4, QA, NLI.
- Search for "readout". Check that each use refers to the final unembedding / LM head interface, not the whole model or generated behavior.
- Search for "model behavior", "sequence-level", "generation", and "task". Verify the paper does not claim SRP directly explains these without an explicit linear proxy or additional evaluation.

## 11. Open Terminology Questions

### Resolved Decisions for Submission

- Use **local score decomposition** in prose and figure captions; use **local score decomposition** in formal Method prose when referring to the mathematical identity.
- Define **scalar readout target** on first use, then use **readout target** once the scalar and linear scope is clear.
- Keep **projection profile** out of the main technical spine. Use it only as display shorthand after defining **readout feature projection vector**.
- Pair **readout feature** with **SAE decoder direction** at first use in each major section that discusses labels or qualitative examples.
- Use **constrained readout-side edit** in main-text prose. Reserve **feature steering** for appendix headings only if the score-level scope is explicit.
- Avoid **robust** outside "robustness controls" unless the sentence names the tested perturbation, model-family, seed, or tokenization/frequency control.
- Prefer **low-residual**, **sign-preserved**, or **small relative reconstruction error** in qualitative captions rather than bare **faithful**.
- Define **benchmark-derived readout target** before the first benchmark table and state that it measures readout-target reconstruction, not end-to-end task accuracy.
- Keep **reasoning-distilled readout/checkpoint** as a model-suite descriptor only; do not frame any result as explaining reasoning.
- Keep **feature label** as the canonical term, but immediately define it as a row-level summary. Do not rename it to "row-summary label" unless reviewers object.
- Collapse **local competitor margin** and **top-competitor margin** to **top-competitor margin** in definitions and tables. Use "local" only when explaining that the competitor set is chosen for the current hidden state or ranking.
- Use **LM-head replacement fidelity** as the canonical metric name. Use **readout replacement** only as broader prose after "readout" has been defined.

### Remaining Questions

- Should the paper keep both **centered logit** and **reference contrast**, or standardize on **reference contrast** everywhere except when describing raw token-logit centering?
- Should feature labels in the main figures include feature ids visually, so the prose can more cleanly distinguish labels from measured objects?
