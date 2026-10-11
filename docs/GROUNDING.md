# Evidence assessment, grounding, and citations

## Decision boundary

`EvidenceAssessor` runs after authorized retrieval and before generation. It checks:

1. whether any evidence exists;
2. the strategy-specific minimum top relevance signal;
3. lexical coverage between meaningful question terms and the first three evidence passages;
4. close-scoring versions of the same policy for explicitly current questions.

Failure produces an abstention reason such as `NO_RETRIEVAL_RESULTS`, `LOW_RELEVANCE_SCORE`,
`LOW_QUERY_EVIDENCE_OVERLAP`, or `CONTRADICTORY_VERSION_EVIDENCE`. These checks are transparent,
conservative heuristics. They are not probability calibration and cannot reliably recognize every
paraphrase, contradiction, incomplete multi-document answer, or out-of-domain question.

The policy cutoffs were selected using the 8-case development calibration file. The held-out file
contains 15 different case IDs. This separation avoids directly optimizing cutoffs on reported
labels, but both sets are small, synthetic, authored against the same corpus, and therefore remain
vulnerable to design leakage and large statistical uncertainty.

## Generation

Offline mode deterministically extracts the highest query-term-overlap sentence from up to three
evidence chunks and attaches that chunk's stable ID. Chunks with no question-term overlap are
skipped. It makes tests reproducible and demonstrates the response
contract; it is not abstractive question answering and can include relevant but non-responsive
sentences.

Optional Gemini mode receives only authorized evidence and is instructed to return JSON containing
an answer and structured claims with exact chunk IDs. Empty output, malformed JSON, API failures,
or unsupported citations result in an abstention rather than an uncited fallback answer.

## What citation validation means

For every factual claim, validation checks that:

- at least one chunk ID is present; and
- every cited chunk ID occurs in the retrieved, authorization-filtered context for that request.

Citation validity does not show that the cited passage entails the claim, that the claim correctly
interprets the passage, that all necessary evidence was retrieved, or that the underlying document
is current. `claim_evidence_coverage` measures citation presence only. Factual correctness and
entailment require a separate evaluation methodology that is not implemented in Phase 4.

## Failure handling

The service fails closed to a structured no-answer response for insufficient evidence, generator
failure, empty generated answers, missing citations, and citations outside the authorized context.
Responses retain retrieved evidence and assessment signals for diagnosis, while claims are omitted
when generation cannot be trusted.
