"""보고서 마크다운의 렌더링·참조 오류 검사.

이 세션에서 실제로 두 번 이상 재발한 오류들을 고정해 둔 것이다. 본문을 고친 뒤
    python -m src.check_report 최종보고서_v1.md
로 돌린다. 종료 코드가 0이 아니면 문제가 남아 있다.

검사 항목
  1. 붙여쓴 물결표      "0.8~0.9"의 ~ 두 개가 짝을 이뤄 사이 문장에 취소선이 그어진다.
  2. 인라인 수식 속 별표  "$\\lambda^*$"의 *가 마크다운 강조로 먹혀 수식이 깨진다.
  3. TeX 간격 매크로     "\\;" "\\," 는 마크다운이 이스케이프로 먼저 소비해 ";" ","로 새어 나온다.
  4. 절 참조 유효성      "2.3.7절" 처럼 존재하지 않는 절을 가리키는 참조.
  5. 그림 번호 순서      캡션 번호가 문서 등장 순서와 어긋나는 경우.
  6. 이미지 경로         참조된 PNG가 실제로 없는 경우.
"""
import re
import sys
from pathlib import Path


def check(path: Path):
    s = path.read_text()
    lines = s.split("\n")
    problems = []

    # 1. 붙여쓴 물결표 (양옆에 공백이 없으면 GFM이 취소선 구분자로 읽는다)
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"[^ ~]~[^ ~]", l):
            problems.append((i, "취소선 유발", f"붙여쓴 ~ : …{l[max(0,m.start()-12):m.end()+12]}… → en dash(–) 권장"))

    # 2. 인라인 수식 안의 리터럴 별표.
    #    $로 쪼갠 뒤 홀수 조각만이 수식 내부다. 굵게(**) 표기가 수식 사이에 끼어도
    #    오탐이 나지 않도록 이렇게 센다.
    for i, l in enumerate(lines, 1):
        parts = l.split("$")
        if len(parts) % 2 == 0:          # $ 개수가 홀수면 수식이 열린 채 끝난 줄
            continue
        for inside in parts[1::2]:
            if "*" in inside:
                problems.append((i, "수식 강조 충돌", f"인라인 수식 속 * : ${inside[:40]}$ → \\ast 권장"))

    # 2b. 따옴표·괄호에 바로 붙은 인라인 수식 구분자.
    #     GitHub은 여는 $ 앞에 "  ' ( 같은 문자가 바로 붙으면 수식 시작으로 보지 않고,
    #     그 여파로 같은 줄 뒤쪽의 짝까지 어긋나 줄 전체가 원문 그대로 노출된다.
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"[\"'“”‘’(\[]\$[^ $]", l):
            problems.append((i, "수식 구분자 인접", f"…{m.group(0)}… → 여는 $ 앞의 문자를 떼거나 문장을 고쳐 쓸 것"))

    # 3. 마크다운이 삼켜버리는 TeX 매크로.
    #    백슬래시 뒤 ASCII 구두점은 마크다운이 이스케이프로 먼저 소비하므로
    #    수식 렌더러에 닿기 전에 사라진다(\; \, \! \: 등).
    for i, l in enumerate(lines, 1):
        for m in re.finditer(r"\\[;,!:]", l):
            problems.append((i, "간격 매크로", f"{m.group(0)} 사용 → 일반 공백이나 \\ 로 대체"))

    # 4. 절 참조 유효성
    heads = {m.group(1) for l in lines
             if (m := re.match(r"#+ (\d+(?:\.\d+)*)\.", l))}
    for i, l in enumerate(lines, 1):
        for ref in re.findall(r"(\d+\.\d+(?:\.\d+)?)절", l):
            if ref not in heads:
                problems.append((i, "없는 절 참조", f"{ref}절"))

    # 5. 그림 번호가 등장 순서와 일치하는지
    caps = [(i, int(m.group(1))) for i, l in enumerate(lines, 1)
            if (m := re.match(r"!\[그림 (\d+)\.", l))]
    nums = [n for _, n in caps]
    if nums != sorted(nums):
        problems.append((caps[0][0] if caps else 0, "그림 순서", f"등장 순서 {nums}"))

    # 6. 이미지 경로 실재
    root = path.parent
    for i, l in enumerate(lines, 1):
        for rel in re.findall(r"!\[[^\]]*\]\(([^)]+\.png)\)", l):
            if not (root / rel).exists():
                problems.append((i, "깨진 이미지", rel))

    return problems


def main(argv):
    targets = [Path(a) for a in argv[1:]] or [Path("최종보고서_v1.md")]
    total = 0
    for p in targets:
        if not p.exists():
            print(f"{p}: 파일 없음")
            total += 1
            continue
        problems = check(p)
        total += len(problems)
        print(f"\n{p} — {'이상 없음' if not problems else f'{len(problems)}건'}")
        for line, kind, detail in problems:
            print(f"  L{line:<5} [{kind}] {detail}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
