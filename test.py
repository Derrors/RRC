from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('./bert-pretrained-model/bert-base-uncased')
string = "The So called laptop Runs to Slow and I hate it ! Do not\u00a0 buy it ! It is the worst laptop ever"
s1 = ' '.join(tokenizer.tokenize(string))
s2 = ' '.join(tokenizer.tokenize("worst"))
start = s1.find(s2, 76)
end = start + len(s2)

char2id = dict()
i = 0
for idx, c in enumerate(s1):
    if c == ' ' or c == '\t' or c == '\r' or c == '\n':
        i += 1
    else:
        char2id[idx] = i
print(s1)
print(char2id[start])
print(char2id[end - 1])
