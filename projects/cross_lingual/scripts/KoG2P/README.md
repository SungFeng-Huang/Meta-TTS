# KoG2P
Given an input of a series of Korean graphemes/letters (i.e. Hangul), KoG2P outputs the corresponding pronunciations.

한국어의 문자열로부터 발음열을 생성하는 파이썬 기반 G2P 패키지입니다.  
터미널에서 원하는 문자열을 함께 입력해 사용할 수 있습니다.

## How to use?
On terminal, you simply can type in your input within quotations:

	$ python g2p.py '박물관'

Then you'll get /방물관/ symbolized as follows:

	p0 aa ng mm uu ll k0 wa nf

NB. Your input does not necessarily need to be a lemma or a legitimate sequence of Korean; the system will provide an output based on the phonological rules of Korean for any sequences in Hangul.

  
## Requirement
- Python 2.7 or 3.x

  
## Symbol table
Please check out the symbol table below for the mapping.

| C/V       | Position    | Symbols in Hangul | Symbols in KoG2P |
|-----------|-------------|-------|-------|
| consonant | onset       | ㅂ    | p0    |
| consonant | onset       | ㅍ    | ph    |
| consonant | onset       | ㅃ    | pp    |
| consonant | onset       | ㄷ    | t0    |
| consonant | onset       | ㅌ    | th    |
| consonant | onset       | ㄸ    | tt    |
| consonant | onset       | ㄱ    | k0    |
| consonant | onset       | ㅋ    | kh    |
| consonant | onset       | ㄲ    | kk    |
| consonant | onset       | ㅅ    | s0    |
| consonant | onset       | ㅆ    | ss    |
| consonant | onset       | ㅎ    | h0    |
| consonant | onset       | ㅈ    | c0    |
| consonant | onset       | ㅊ    | ch    |
| consonant | onset       | ㅉ    | cc    |
| consonant | onset       | ㅁ    | mm    |
| consonant | onset       | ㄴ    | nn    |
| consonant | onset       | ㄹ    | rr    |
| consonant | coda        | ㅂ    | pf    |
| consonant | coda        | ㅍ    | ph    |
| consonant | coda        | ㄷ    | tf    |
| consonant | coda        | ㅌ    | th    |
| consonant | coda        | ㄱ    | kf    |
| consonant | coda        | ㅋ    | kh    |
| consonant | coda        | ㄲ    | kk    |
| consonant | coda        | ㅅ    | s0    |
| consonant | coda        | ㅆ    | ss    |
| consonant | coda        | ㅎ    | h0    |
| consonant | coda        | ㅈ    | c0    |
| consonant | coda        | ㅊ    | ch    |
| consonant | coda        | ㅁ    | mf    |
| consonant | coda        | ㄴ    | nf    |
| consonant | coda        | ㅇ    | ng    |
| consonant | coda        | ㄹ    | ll    |
| consonant | coda        | ㄱㅅ  | ks    |
| consonant | coda        | ㄴㅈ  | nc    |
| consonant | coda        | ㄴㅎ  | nh    |
| consonant | coda        | ㄹㄱ  | lk    |
| consonant | coda        | ㄹㅁ  | lm    |
| consonant | coda        | ㄹㅂ  | lb    |
| consonant | coda        | ㄹㅅ  | ls    |
| consonant | coda        | ㄹㅌ  | lt    |
| consonant | coda        | ㄹㅍ  | lp    |
| consonant | coda        | ㄹㅎ  | lh    |
| consonant | coda        | ㅂㅅ  | ps    |
| vowel     | monophthong | ㅣ    | ii    |
| vowel     | monophthong | ㅔ    | ee    |
| vowel     | monophthong | ㅐ    | qq    |
| vowel     | monophthong | ㅏ    | aa    |
| vowel     | monophthong | ㅡ    | xx    |
| vowel     | monophthong | ㅓ    | vv    |
| vowel     | monophthong | ㅜ    | uu    |
| vowel     | monophthong | ㅗ    | oo    |
| vowel     | diphthong   | ㅖ    | ye    |
| vowel     | diphthong   | ㅒ    | yq    |
| vowel     | diphthong   | ㅑ    | ya    |
| vowel     | diphthong   | ㅕ    | yv    |
| vowel     | diphthong   | ㅠ    | yu    |
| vowel     | diphthong   | ㅛ    | yo    |
| vowel     | diphthong   | ㅟ    | wi    |
| vowel     | diphthong   | ㅚ    | wo    |
| vowel     | diphthong   | ㅙ    | wq    |
| vowel     | diphthong   | ㅞ    | we    |
| vowel     | diphthong   | ㅘ    | wa    |
| vowel     | diphthong   | ㅝ    | wv    |
| vowel     | diphthong   | ㅢ    | xi    |
  
NB. IPA symbols for Korean phones can be found in the following page: [IPA for Korean](https://en.wikipedia.org/wiki/Help:IPA_for_Korean).   

## Reference
Please cite the following if using this code:

	@misc{cho2017kog2p,
	  title = {Korean Grapheme-to-Phoneme Analyzer (KoG2P)},
	  author = {Yejin Cho},
	  year = {2017},
	  publisher = {GitHub},
	  journal = {GitHub repository},
	  howpublished = {\url{https://github.com/scarletcho/KoG2P}}
	}

## Thank you for your citations!

- Yoon Seok Hong, Kyung Seo Ki, and Gahgene Gweon. 2018. Automatic Miscue Detection Using RNN Based Models with Data Augmentation. In Proc. Interspeech 2018. 1646-1650. [[pdf](https://www.isca-speech.org/archive/Interspeech_2018/pdfs/1644.pdf)]

- Younggun Lee and Taesu Kim. 2018. Learning pronunciation from a foreign language in speech synthesis network. arXiv preprint. arXiv:1811.09364. [[pdf](https://arxiv.org/pdf/1811.09364v1.pdf)]
