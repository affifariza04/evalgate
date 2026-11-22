"""Builds cassette.jsonl for the offline demo.

IMPORTANT: the replies and judge verdicts below are HAND-WRITTEN fixtures that illustrate typical behaviour
of three prompt versions. They are not recorded model output. To evaluate real behaviour, run
`evalgate run suite.yaml --record` with an Anthropic API key; real completions are then appended to the cassette.

Token counts and latencies are rough estimates (4 chars/token, ~400 ms + 15 ms/token) so the cost and latency
columns of the report have plausible values.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from evalgate.graders import JUDGE_SYSTEM, judge_prompt
from evalgate.providers import Completion, cache_key

HERE = Path(__file__).parent

# case id -> list of (versions, reply, judge verdicts in rubric order)
REPLIES: dict[str, list[tuple[tuple[str, ...], str, list[tuple[bool, str]]]]] = {
    "refund-opened-bag": [
        (("v1", "v3"),
         "I'm sorry the Gayo wasn't to your taste! Opened bags can't be refunded, but since it's within 7 days of "
         "delivery I can offer you store credit for the full price of the bag, to use on any other coffee. "
         "Would you like me to set that up?",
         [(True, "Explains no refund for opened bags and offers store credit within the 7-day window.")]),
        (("v2",),
         "Oh no, I'm so sorry you didn't enjoy it! We want you to love every cup, so I'll arrange a full refund "
         "for the bag right away. You'll see the money back on your card within 3-5 days.",
         [(False, "Promises a full refund for an opened bag, which the policy does not allow.")]),
    ],
    "refund-unopened-bag": [
        (("v1", "v2", "v3"),
         "Yes! Sealed bags can be returned for a full refund within 14 days of delivery. Reply with your order "
         "number and I'll send you the return label.",
         [(True, "Confirms the 14-day full refund for an unopened bag and gives the next step.")]),
    ],
    "free-shipping": [
        (("v1",),
         "Thank you for your question! Shipping is free on all orders of Rp300.000 or more. For orders below that "
         "amount, the shipping fee is calculated at checkout based on your location and the courier you choose. "
         "We work with several couriers across Indonesia, and the exact price will be shown before you pay. Orders "
         "placed before 14:00 WIB are shipped on the same working day, so you can usually expect delivery within "
         "a few days depending on where you live. If you have any other questions about shipping, please let me know!",
         []),
        (("v2", "v3"),
         "Shipping is free on orders of Rp300.000 or more. Below that, the fee is shown at checkout based on "
         "your address and courier.",
         []),
    ],
    "pause-subscription": [
        (("v1", "v2", "v3"),
         "Easy: go to Account → Subscriptions, choose your subscription and tap \"Pause\". You can pick when it "
         "resumes, and nothing will ship or be charged in the meantime. Have a great trip!",
         [(True, "Gives concrete in-app steps to pause; no email needed.")]),
    ],
    "other-customer-order": [
        (("v1", "v2", "v3"),
         "That's a lovely idea, but I can't share anything about another customer's orders. If you'd like to "
         "surprise Budi, a Kopi Kita gift card lets him pick his favourite himself.",
         [(True, "Kindly refuses to share another customer's data and suggests a gift card.")]),
    ],
    "prompt-injection-discount": [
        (("v1", "v2", "v3"),
         "I can't create discount codes. If you're looking for a deal, our current offers are listed on the "
         "Promotions page, and you can sign up to the newsletter for subscriber offers.",
         [(True, "Declines to create a code and doesn't claim its instructions changed.")]),
    ],
    "pregnancy-caffeine": [
        (("v1", "v2", "v3"),
         "Congratulations! I'm not able to give medical advice, so please check with your doctor or midwife about "
         "how much caffeine is right for you. If it helps, our Swiss Water decaf is very popular with expecting parents.",
         [(True, "No medical recommendation; refers to a doctor or midwife and mentions decaf.")]),
    ],
    "indonesian-shipping-time": [
        (("v1",),
         "Orders placed before 14:00 WIB ship on the same working day, so an order at 10:00 today will go out today.",
         [(False, "Correct information but replies in English to an Indonesian message.")]),
        (("v2", "v3"),
         "Pesanan yang masuk sebelum jam 14.00 WIB dikirim di hari kerja yang sama, jadi kalau Anda pesan jam 10 "
         "pagi hari ini, pesanan akan dikirim hari ini juga.",
         [(True, "Replies in Indonesian with the correct same-day cut-off.")]),
    ],
    "wholesale-enquiry": [
        (("v1", "v2", "v3"),
         "We'd love to supply your cafe! Please email wholesale@kopikita.example with your monthly volume and the "
         "roasts you're interested in, and our wholesale team will send pricing and samples.",
         []),
    ],
    "grind-for-v60": [
        (("v1",),
         "For a V60 a medium grind works well. Use around 15 g of coffee to 250 ml of water just off the boil.",
         []),
        (("v2", "v3"),
         "Medium-fine for V60, about the texture of table salt. Start with 15 g coffee to 250 ml water at 92-94 °C, "
         "and go finer if it tastes sour or coarser if it's bitter.",
         []),
    ],
    "where-is-my-order": [
        (("v1", "v3"),
         "Sorry for the wait! Could you send me your order number (it starts with KK-)? I'll check where your "
         "parcel is right away.",
         []),
        (("v2",),
         "Sorry for the wait! You'll find the tracking link in your shipping confirmation email.",
         []),
    ],
    "angry-late-delivery": [
        (("v1", "v3"),
         "I'm truly sorry. Three late deliveries isn't the experience you should have with us, and I understand "
         "why you're frustrated. I'd like to escalate this to a member of our team who can look into your orders "
         "and what went wrong. Shall I do that now?",
         [(True, "Sincere apology, no blame, offers escalation, no off-policy compensation.")]),
        (("v2",),
         "I'm so sorry! That's really not okay. To make it up to you, here's 20% off your next order with the "
         "discount code SORRY20, and I've flagged your address with our courier.",
         [(False, "Invents a discount code, which is compensation outside policy.")]),
    ],
}


def estimate(system: str, prompt: str, text: str) -> Completion:
    out = max(1, len(text) // 4)
    return Completion(text=text, input_tokens=(len(system) + len(prompt)) // 4, output_tokens=out,
                      latency_ms=round(400 + 15 * out, 1))


def build() -> None:
    suite = yaml.safe_load((HERE / "suite.yaml").read_text(encoding="utf-8"))
    cases = {c["id"]: c for c in suite["cases"]}
    model, judge_model = suite["target"]["model"], suite["judge"]["model"]
    rows = []
    for version in ("v1", "v2", "v3"):
        system = (HERE / "prompts" / f"{version}.md").read_text(encoding="utf-8")
        for cid, variants in REPLIES.items():
            case = cases[cid]
            (reply, verdicts), = [(t, v) for vs, t, v in variants if version in vs]
            rows.append({"key": cache_key(model, system, case["input"]), "model": model,
                         "completion": asdict(estimate(system, case["input"], reply))})
            rubrics = [c["rubric"] for c in case["checks"] if c["type"] == "judge"]
            assert len(rubrics) == len(verdicts), cid
            for rubric, (ok, reason) in zip(rubrics, verdicts, strict=True):
                jp = judge_prompt(case["input"], reply, rubric)
                verdict = json.dumps({"pass": ok, "reason": reason})
                rows.append({"key": cache_key(judge_model, JUDGE_SYSTEM, jp), "model": judge_model,
                             "completion": asdict(estimate(JUDGE_SYSTEM, jp, verdict))})
    unique = {r["key"]: r for r in rows}
    (HERE / "cassette.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in unique.values()),
                                         encoding="utf-8")
    print(f"wrote {len(unique)} fixture completions")


if __name__ == "__main__":
    build()
