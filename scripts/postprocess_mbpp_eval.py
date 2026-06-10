#!/usr/bin/env python3
import argparse
import json
import pathlib
import traceback
import signal
from typing import Tuple, Optional


# ========================
#  超时相关
# ========================
class TimeoutException(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutException()


# ========================
#  工具函数
# ========================
def extract_first_block(gen) -> str:
    """
    从生成中提取真正的代码：
    - 兼容 gen 为 str / list / 嵌套 list / None
    - 按第一个 [DONE] 截断
    - 去掉首尾空白
    """
    # 1) 规整成字符串
    if gen is None:
        gen = ""
    elif isinstance(gen, list):
        # 下钻直到拿到非 list 的元素；空 list 则变成 ""
        while isinstance(gen, list):
            if len(gen) == 0:
                gen = ""
                break
            gen = gen[0]
        if gen is None:
            gen = ""
    elif not isinstance(gen, str):
        gen = str(gen)

    # 2) 截断 [DONE]
    if "[DONE]" in gen:
        gen = gen.split("[DONE]", 1)[0]

    return gen.strip()


def get_generation(sample: dict) -> str:
    """
    从 sample 中拿到模型的生成文本。
    优先用 filtered_resps，否则退回 resps。
    兼容字段为 str / list[str] / list[list[str]] / 更深嵌套。
    """
    if sample.get("filtered_resps") is not None:
        gen = sample["filtered_resps"]
    elif sample.get("resps") is not None:
        gen = sample["resps"]
    else:
        raise ValueError("No generation field found in sample (no 'filtered_resps' or 'resps').")

    # 统一下钻：filtered_resps 可能是 ["..."] 或 [["..."]]；resps 也可能是嵌套
    while isinstance(gen, list):
        if len(gen) == 0:
            gen = ""
            break
        gen = gen[0]

    if gen is None:
        return ""
    if isinstance(gen, str):
        return gen
    return str(gen)



def run_mbpp_tests(sample: dict, verbose: bool = False) -> Tuple[bool, Optional[str]]:
    """
    对单条 MBPP sample 重新跑测试：
    - 返回 (是否通过, 错误信息或 None)
    """
    doc = sample["doc"]
    tests = list(doc.get("test_list", [])) + list(doc.get("challenge_test_list", []))
    setup_code = doc.get("test_setup_code", "") or ""

    try:
        gen_text = get_generation(sample)
    except Exception as e:
        return False, f"no_generation: {e}"

    code = extract_first_block(gen_text)

    # 为安全起见，每个样本单独的执行环境
    glb: dict = {}

    try:
        if setup_code.strip():
            exec(setup_code, glb, glb)
        if code.strip():
            exec(code, glb, glb)
        # 依次执行所有断言
        for t in tests:
            if t.strip():
                exec(t, glb, glb)
    except Exception as e:
        if verbose:
            print(f"[Doc {sample.get('doc_id')}] failed with error:\n{traceback.format_exc()}")
        return False, repr(e)

    return True, None


# ========================
#  主逻辑
# ========================
def main():
    parser = argparse.ArgumentParser(
        description="Postprocess MBPP samples (truncate at [DONE] and re-evaluate pass@1)."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入的 JSONL 样本文件（如 samples_mbpp_*.jsonl）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出截断后样本的 JSONL 文件路径（可选）",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个失败样本的错误栈",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="每个样本的执行时间上限（秒），<=0 则不启用超时，默认 5 秒",
    )
    args = parser.parse_args()

    in_path = pathlib.Path(args.input)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input file not found: {in_path}")

    out_f = None
    if args.output is not None:
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    total = 0
    passed = 0

    # 只在类 Unix 系统上可用；你现在在 Ubuntu，可以用
    signal.signal(signal.SIGALRM, _timeout_handler)

    with in_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue

            sample = json.loads(line)
            # 有些文件可能没有 doc_id，这里兜底用行号
            if "doc_id" not in sample:
                sample["doc_id"] = sample.get("doc", {}).get("task_id", line_idx)

            total += 1

            # === 每个样本加超时保护 ===
            try:
                if args.timeout and args.timeout > 0:
                    signal.alarm(int(args.timeout))
                ok, err = run_mbpp_tests(sample, verbose=args.verbose)
            except TimeoutException:
                ok = False
                err = f"Timeout (> {args.timeout}s)"
                if args.verbose:
                    print(f"[Doc {sample.get('doc_id')}] failed with error: {err}")
            finally:
                if args.timeout and args.timeout > 0:
                    signal.alarm(0)

            if ok:
                passed += 1

            # 如果需要，把生成截断后写回新的 JSONL
            if out_f is not None:
                try:
                    gen_text = get_generation(sample)
                    truncated = extract_first_block(gen_text)
                    # 只覆盖 filtered_resps，保留其他字段不变
                    sample["filtered_resps"] = [truncated]
                except Exception:
                    # 出问题就保持原样
                    pass
                out_f.write(json.dumps(sample, ensure_ascii=False))
                out_f.write("\n")

            # 简单进度提示（可选）
            if args.verbose and total % 50 == 0:
                print(f"[Progress] processed {total} samples...")

    if out_f is not None:
        out_f.close()

    if total == 0:
        print("No valid samples found.")
    else:
        print(f"Total samples: {total}")
        print(f"Passed: {passed}")
        print(f"pass_at_1 (visible tests): {passed / total:.4f}")


if __name__ == "__main__":
    main()
