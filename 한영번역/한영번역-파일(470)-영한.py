from transformers import pipeline

# 모델: en→ko  (※ en-ko의 정식 업스트림 모델은 tc-big 계열을 사용)
en2ko = pipeline("translation", model="Helsinki-NLP/opus-mt-tc-big-en-ko")

input_file = "input_en.txt"
output_file = "output_ko.txt"

with open(input_file, "r", encoding="utf-8") as f:
    text = f.read()

translated = en2ko(text)[0]["translation_text"]

print("===== 🙂 번역 결과 (EN→KO) =====")
print(translated)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(translated)
