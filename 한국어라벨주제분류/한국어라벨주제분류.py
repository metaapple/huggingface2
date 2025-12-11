from transformers import pipeline

# 제로샷 분류 파이프라인
zero = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# 긴 문장 (문화 콘텐츠 관련)
text = """
최근에 개봉한 한국 영화는 사회적 문제를 날카롭게 비판하면서도 따뜻한 가족애를 동시에 보여주고 있습니다.
특히 청년 세대가 겪는 취업난과 세대 갈등을 섬세하게 다루었고,
OST 음악은 대중적으로 큰 인기를 끌고 있으며 드라마화 제안까지 이어지고 있습니다.
"""

# 라벨 후보 (문화 콘텐츠 카테고리)
labels = [
    "영화", "드라마", "음악", "문학", "게임",
    "패션", "미술", "무용", "웹툰", "예능"
]

# 제로샷 분류 수행
res = zero(
    text,
    candidate_labels=labels,
    hypothesis_template="이 문장은 {}와 관련이 있다."
)

print(res)
# 이 문장은 여러 라벨 중 가장 높은 가능성에 해당하는 '드라마'과(와) 관련이 있다. (score=0.1416)