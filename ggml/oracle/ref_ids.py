#!/usr/bin/env python3
"""Generate reference tokenizations for the Zig tokenizer port.

The first section rebuilds the three oracle sequences end to end from the layout rules in
``laya/vlm.py`` -- no transformers, no torch -- and checks them against the ``input_ids.i32``
the oracle dumped from the real model. That is the acceptance test for the layout half of the
port. The rest writes a corpus of (text, ids, pre-tokenizer split) rows for the BPE half.

Usage: ref_ids.py <model_dir> <oracle_out_dir> <out_dir>
"""

import array
import json
import struct
import os
import random
import sys

from tokenizers import Tokenizer

PREFIX_TEXT = "<|im_start|>User:"
QUESTION_TEXT = "\n%s question: %s<end_of_utterance>\nAssistant: Options:\n"
OPTION_BULLET = "- "
OPTION_END = "\n"

FAKE = "<fake_token_around_image>"
GLOB = "<global-img>"
IMAGE = "<image>"


def render_options(q):
    """laya.common.render_options, verbatim."""
    t, crit = q["t"], q.get("crit")
    if t == "choice":
        return [k if not v else "%s: %s" % (k, v) for k, v in crit.items()]
    if t == "score":
        return ["level %d: %s" % (i, c) for i, c in enumerate(crit)]
    crit = crit or {}
    return [
        "false: " + (crit.get("false") or "no, the statement does not hold"),
        "true: " + (crit.get("true") or "yes, the statement holds"),
    ]


CASES = {
    "choice": {
        "type": "choice",
        "instructions": "What setting does the image show?",
        "criteria": {
            "indoor": "Indoors",
            "outdoor": "Outdoors",
            "unknown": "Unclear",
            "abstract": "An abstract pattern",
        },
    },
    "noul": {
        "type": "noul",
        "instructions": "Is there a bright rectangular block in the image?",
    },
    "score": {
        "type": "score",
        "instructions": "How much readable text is in the image?",
        "criteria": ["None", "A few words", "Many words"],
    },
}

STATE_TEXT = json.dumps({"note": "synthetic fixture, deterministic"}, ensure_ascii=False)


def to_internal(qdef):
    """laya.vlm.VLMAgent._to_internal, verbatim."""
    t = qdef["type"]
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef["instructions"]
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def build_sequence(tok, q, image_seq_len=64, max_len=1024, head_max_len=256, n_images=1):
    """laya.vlm.build_vlm_inputs, specialised to this project's config (no option permutations)."""
    def enc(s):
        return tok.encode(s, add_special_tokens=False).ids

    expansion = (FAKE + GLOB + IMAGE * image_seq_len + FAKE) * n_images
    prefix_ids = enc(PREFIX_TEXT + expansion)

    end_id = enc(OPTION_END)
    assert len(end_id) == 1, "option terminator must be a single token"
    end_id = end_id[0]

    opts = render_options(q)
    opt_ids = [enc(OPTION_BULLET + opts[i].replace(OPTION_END, " "))[:48] for i in range(len(opts))]
    head_ids = enc(QUESTION_TEXT % (q["t"], str(q["ins"]).replace("<end_of_utterance>", " ")))

    opt_budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)) - 1)
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) + 1 for o in opt_ids)
    if len(head_ids) > max(8, opt_budget):
        keep = max(8, opt_budget)
        head_ids = head_ids[: keep // 2] + head_ids[-(keep - keep // 2):]

    tail = list(head_ids)
    markers = []
    span_start = len(tail)
    for o in opt_ids:
        tail.extend(o)
        tail.append(end_id)
        markers.append(len(tail) - 1)

    room = max(0, max_len - len(prefix_ids) - len(tail))
    st = enc(STATE_TEXT)[:room]
    off = len(prefix_ids) + len(st)
    return {
        "ids": prefix_ids + st + tail,
        "markers": [m + off for m in markers],
        "option_span": (span_start + off, len(tail) + off),
    }



def emit_binary(rows, path):
    """A format with no parsing ambiguity for the Zig harness.

    magic 'TOKR', u32 n_rows, then per row:
      u32 label_len, label bytes
      u32 text_len,  text utf-8 bytes
      u32 n_ids,     i32 ids
      u32 n_pieces,  per piece: u32 len, bytes, u32 n_ids, i32 ids
    All little-endian.
    """
    buf = bytearray(b"TOKR")
    buf += struct.pack("<I", len(rows))
    for label, text, ids, pieces in rows:
        lb, tb = label.encode(), text.encode()
        buf += struct.pack("<I", len(lb)) + lb
        buf += struct.pack("<I", len(tb)) + tb
        buf += struct.pack("<I", len(ids)) + struct.pack("<%di" % len(ids), *ids)
        buf += struct.pack("<I", len(pieces))
        for piece, pids in pieces:
            pb = piece.encode()
            buf += struct.pack("<I", len(pb)) + pb
            buf += struct.pack("<I", len(pids)) + struct.pack("<%di" % len(pids), *pids)
    with open(path, "wb") as f:
        f.write(buf)
    return len(buf)


def unicode_probes():
    """`a<cp>b` for a systematic sample of codepoints.

    The regex splits on `\\p{L}`/`\\p{N}`/`\\s`, so a port needs Unicode category tables. These rows
    make the tables checkable: the shape of the split for `a<cp>b` reveals which class `<cp>` fell in
    (letter/number -> one merged piece, space -> its own piece, other -> its own piece with the
    non-space class). Anything the tables get wrong shows up here rather than in production.
    """
    cps = list(range(0x00, 0x100))                      # Latin-1, the dense edge cases
    cps += list(range(0x100, 0x3000, 64))
    cps += list(range(0x3000, 0xA000, 64))
    cps += list(range(0x4E00, 0xA000, 127))             # CJK
    cps += list(range(0xAC00, 0xD7A4, 64))              # Hangul syllables
    cps += list(range(0xA000, 0x11000, 64))
    cps += list(range(0x1F300, 0x1FB00, 16))            # emoji blocks
    cps += list(range(0x10000, 0x110000, 4096))
    out = []
    seen = set()
    for cp in cps:
        if 0xD800 <= cp <= 0xDFFF or cp in seen:
            continue
        seen.add(cp)
        out.append(("probe/U+{:04X}".format(cp), "a" + chr(cp) + "b"))
    return out

def corpus():
    """Strings the Zig tokenizer has to get right, grouped by what they stress."""
    cases = []
    for name, qdef in CASES.items():
        q = to_internal(qdef)
        opts = render_options(q)
        cases += [
            (name + "/prefix", PREFIX_TEXT + FAKE + GLOB + IMAGE * 64 + FAKE),
            (name + "/state", STATE_TEXT),
            (name + "/head", QUESTION_TEXT % (q["t"], str(q["ins"]).replace("<end_of_utterance>", " "))),
        ]
        for i, o in enumerate(opts):
            cases.append((name + "/opt%d" % i, OPTION_BULLET + o.replace("\n", " ")))

    cases += [
        ("empty", ""),
        ("space", " "),
        ("two-spaces", "a  b"),
        ("three-spaces", "a   b"),
        ("lead-space", " hello"),
        ("trail-space", "hello "),
        ("newline", "\n"),
        ("newlines", "\n\n\n"),
        ("tab", "\ta\tb"),
        ("crlf", "a\r\nb"),
        ("contraction-s", "it's"),
        ("contraction-t", "don't"),
        ("contraction-re", "we're"),
        ("contraction-ve", "i've"),
        ("contraction-m", "i'm"),
        ("contraction-ll", "we'll"),
        ("contraction-d", "i'd"),
        ("apostrophe-alone", "'"),
        ("apostrophe-x", "'x"),
        ("digits", "12345"),
        ("digit-group", "a 123 b"),
        ("mixed", "abc123def"),
        ("punct-run", "!!!???..."),
        ("punct-space", "a !!! b"),
        ("caps", "HELLO WORLD"),
        ("json", STATE_TEXT),
        ("url", "https://example.com/a?b=c#d"),
        ("path", "/usr/local/bin/python3.12"),
        ("numbers", "3.14159 -2e10 0x1F"),
        ("cjk", "\u8fd9\u662f\u4e00\u4e2a\u6d4b\u8bd5"),
        ("cjk-mixed", "hello \u4e16\u754c 123"),
        ("japanese", "\u3053\u3093\u306b\u3061\u306f\u4e16\u754c"),
        ("korean", "\uc548\ub155\ud558\uc138\uc694"),
        ("accented", "naive cafe resume"),
        ("greek", "\u03b1\u03b2\u03b3\u03b4"),
        ("cyrillic", "\u043f\u0440\u0438\u0432\u0435\u0442"),
        ("emoji", "ok \U0001f44d done"),
        ("brackets", "[1, 2, 3]"),
        ("braces", '{"a": {"b": [true, false, null]}}'),
        ("arith", "What is 2+2?"),
        ("question", "What setting does the image show?"),
        ("quote", 'he said "hi"'),
        ("backslash", "a\\b"),
        ("tabs-mixed", "col1\tcol2\tcol3"),
        ("long-word", "antidisestablishmentarianism"),
        ("repeat", "aaaa" * 20),
        ("space-repeat", " " * 8),
        ("dash", "- "),
        ("bullet", "- option one\n"),
        ("im-tokens", "<|im_start|>User:<|im_end|>"),
        ("utt", "a<end_of_utterance>b"),
        ("imgtok", "<fake_token_around_image><image>"),
        ("prototype", "What is the capital of France?"),
        ("multiline", "line one\nline two\nline three"),
        ("highbyte", "\u00ff\u0100\u2028\u2029"),
        ("nbsp", "a\u00a0b"),
        ("tab-only", "\t\t"),
        ("mixed-ws", " \t\n "),
        ("punct-then-word", "...word"),
        ("word-then-punct", "word..."),
        ("quotes", chr(39) * 3 + chr(34) * 3),
    ]

    rng = random.Random(1234)
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?-_'\"()[]{}:;\n\t"
    for i in range(150):
        n = rng.randint(0, 90)
        cases.append(("rand%03d" % i, "".join(rng.choice(alphabet) for _ in range(n))))
    cases += unicode_probes()
    return cases


def main(argv):
    if len(argv) != 4:
        print(__doc__)
        return 2
    model_dir, oracle_dir, out_dir = argv[1:]
    os.makedirs(out_dir, exist_ok=True)

    tok = Tokenizer.from_file(os.path.join(model_dir, "processor", "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()

    ok = True
    for name, qdef in CASES.items():
        got = build_sequence(tok, to_internal(qdef))
        want = array.array("i")
        with open(os.path.join(oracle_dir, name, "input_ids.i32"), "rb") as f:
            want.frombytes(f.read())
        same = list(want) == got["ids"]
        ok = ok and same
        print("ids={:d} vs {:d}  markers={}  {}".format(
            len(got["ids"]), len(want), got["markers"], "MATCH" if same else "DIFFER"))
        if not same:
            for i in range(min(len(got["ids"]), len(want))):
                if got["ids"][i] != want[i]:
                    print("   first difference at {:d}: got {:d}({!r}) want {:d}({!r})".format(
                        i, got["ids"][i], tok.decode([got["ids"][i]]), want[i], tok.decode([want[i]])))
                    break
    print("layout:", "OK" if ok else "MISMATCH")

    n = 0
    rows = []
    with open(os.path.join(out_dir, "tok_ref.jsonl"), "w") as f:
        for label, text in corpus():
            enc = tok.encode(text, add_special_tokens=False)
            pieces = []
            for piece, _offs in tok.pre_tokenizer.pre_tokenize_str(text):
                pieces.append([piece, tok.encode(piece, add_special_tokens=False).ids])
            f.write(json.dumps({"label": label, "text": text, "ids": enc.ids, "pieces": pieces},
                               ensure_ascii=False) + "\n")
            rows.append((label, text, enc.ids, pieces))
            n += 1
    print("wrote {} bytes -> {}".format(emit_binary(rows, os.path.join(out_dir, "tok_ref.bin")),
                                       os.path.join(out_dir, "tok_ref.bin")))
    print("wrote {} corpus rows -> {}".format(n, os.path.join(out_dir, "tok_ref.jsonl")))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
