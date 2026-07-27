# Platform Community Guidelines

Authored, representative community-guidelines document for this demo platform - modeled on the kind of policy real discussion platforms publish, mapped directly onto the classifier's four moderation categories so the RAG layer has concrete clauses to cite.

## Section 1 - Normal content (no action)

Comments that engage in good-faith discussion, criticism, or disagreement - including strongly worded but non-abusive opinions - are welcome. Disagreement with a person, product, or idea is not by itself a policy violation.

## Section 2 - Offensive language (Section 2.1–2.3)

**2.1 - Personal insults.** Content that demeans, belittles, or insults an individual or their contribution (e.g. calling a person or their opinion "stupid," "garbage," "dumb") without targeting a protected identity characteristic is classified as **Offensive**. This content is typically routed for human review rather than auto-removed, since context (sarcasm, in-group banter, heated but legitimate debate) matters.

**2.2 - Profanity and hostility.** Gratuitous profanity directed at another user, without an accompanying threat or identity-based attack, falls under this section.

**2.3 - Escalation risk.** Repeated Offensive-tier content from the same source, or Offensive content combined with high engagement-signal volatility (many downvotes, low upvote ratio), is treated as an escalation signal for closer review.

## Section 3 - Hate speech (Section 3.1–3.3)

**3.1 - Identity-based attacks.** Content that attacks, demeans, or expresses hatred toward a person or group on the basis of race, religion, gender, disability, or other protected characteristics is classified as **Hate Speech**. This includes explicit exclusionary language ("should be kicked out," "these people are terrible") when directed at an identified group.

**3.2 - Dehumanizing language.** Language that denies a group's humanity or moral worth, even without an explicit call to violence, falls under this section.

**3.3 - Platform-signal corroboration.** When the platform's own opaque moderation signals (if_1/if_2) independently flag a comment in the same range associated with historical Hate Speech content, that corroborates - but does not by itself determine - a Hate Speech classification.

## Section 4 - Severe / violent content (Section 4.1–4.2)

**4.1 - Explicit threats.** Content containing explicit threats of violence, death, or serious harm toward a person or group - including obfuscated variants intended to evade filters (e.g. "k1ll," "d34d") - is classified as **Severe/Violent**, the platform's highest-severity category.

**4.2 - Mandatory auto-action.** Severe/Violent content, once classified above the platform's confidence threshold, is auto-actioned (removed/hidden pending review) rather than queued for discretionary human review first, given the immediacy of harm - consistent with the IT Rules 2021 obligation to act on unlawful threat content without delay.

## Section 5 - Appeals

Any user whose content was actioned under Sections 2–4 may appeal. An appeal triggers re-evaluation of the original comment together with the appeal context and the same policy corpus, and produces a new, independently logged decision.
