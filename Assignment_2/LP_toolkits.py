from re import sub, findall
from constants import punctuations, zero_space, diacritics


def normalizer(news):
    
    # #  map all the used character to an uniform one given in simple_mapping in order to homogenize words
    news = sub_alphabets(news)
    # #     1.1 : remove english letters, numbers, special characters and punctuations
    news = sub('[A-Za-z0-9]', ' ', news)
    for ch in punctuations:
        news = news.replace(ch, ' ')
    for ch in diacritics:
        news = news.replace(ch, '')

    return ' '.join(news.split())


# #     based on parsivar's Normalizer().sub_alphabets() function and changed it in the needed way
def sub_alphabets(doc_string):
    # try:
    #     doc_string = doc_string.decode('utf-8')
    # except UnicodeEncodeError:
    #     pass

    #  #    edited parsivar's function based on whats interested in this assignment
    #  #    example: turn 'آسمان' into 'اسمان' as they have similar meaning and we do not consider only the correct written form
    a0 = 'ء|ﺌ|ﺋ'
    b0 = 'ئ'
    c0 = sub(a0, b0, doc_string)
    a1 = r'ﺎ|ﺄ|ٲ|ٱ|إ|ﺍ|أ'
    a11 = r'ﺁ|آ'
    b1 = r'ا'
    b11 = r'آ'
    c11 = sub(a11, b1, c0)
    c1 = sub(a1, b1, c11)
    a2 = r'ﺐ|ﺏ|ﺑ'
    b2 = r'ب'
    c2 = sub(a2, b2, c1)
    a3 = r'ﭖ|ﭗ|ﭙ|ﺒ|ﭘ'
    b3 = r'پ'
    c3 = sub(a3, b3, c2)
    a4 = r'ﭡ|ٺ|ٹ|ﭞ|ٿ|ټ|ﺕ|ﺗ|ﺖ|ﺘ|ﺜ'
    b4 = r'ت'
    c4 = sub(a4, b4, c3)
    a5 = r'ﺙ|ﺛ'
    b5 = r'ث'
    c5 = sub(a5, b5, c4)
    a6 = r'ﺞ|ﺝ|ڃ|ﺠ|ﺟ'
    b6 = r'ج'
    c6 = sub(a6, b6, c5)
    a7 = r'ڃ|ﭽ|ﭼ|ﭻ'
    b7 = r'چ'
    c7 = sub(a7, b7, c6)
    a8 = r'ﺢ|ﺤ|څ|ځ|ﺣ'
    b8 = r'ح'
    c8 = sub(a8, b8, c7)
    a9 = r'ﺥ|ﺦ|ﺨ|ﺧ'
    b9 = r'خ'
    c9 = sub(a9, b9, c8)
    a10 = r'ڏ|ډ|ﺪ|ﺩ'
    b10 = r'د'
    c10 = sub(a10, b10, c9)
    a11 = r'ﺫ|ﺬ|ﻧ'
    b11 = r'ذ'
    c11 = sub(a11, b11, c10)
    a12 = r'ۯ|ڙ|ڗ|ڒ|ڑ|ڕ|ﺭ|ﺮ'
    b12 = r"ر"
    c12 = sub(a12, b12, c11)
    a13 = r"ﺰ|ﺯ"
    b13 = r"ز"
    c13 = sub(a13, b13, c12)
    a14 = r"ﮊ|ﮋ"
    b14 = r"ژ"
    c14 = sub(a14, b14, c13)
    a15 = r"ݭ|ݜ|ﺱ|ﺲ|ښ|ﺴ|ﺳ"
    b15 = r"س"
    c15 = sub(a15, b15, c14)
    a16 = r"ﺵ|ﺶ|ﺸ|ﺷ"
    b16 = r"ش"
    c16 = sub(a16, b16, c15)
    a17 = r"ﺺ|ﺼ|ﺻ"
    b17 = r"ص"
    c17 = sub(a17, b17, c16)
    a18 = r"ﺽ|ﺾ|ﺿ|ﻀ"
    b18 = r"ض"
    c18 = sub(a18, b18, c17)
    a19 = r"ﻁ|ﻂ|ﻃ|ﻄ"
    b19 = r"ط"
    c19 = sub(a19, b19, c18)
    a20 = r"ﻆ|ﻇ|ﻈ"
    b20 = r"ظ"
    c20 = sub(a20, b20, c19)
    a21 = r"ڠ|ﻉ|ﻊ|ﻌ|ﻋ"
    b21 = r"ع"
    c21 = sub(a21, b21, c20)
    a22 = r"ﻎ|ۼ|ﻍ|ﻐ|ﻏ"
    b22 = r"غ"
    c22 = sub(a22, b22, c21)
    a23 = r"ﻒ|ﻑ|ﻔ|ﻓ"
    b23 = r"ف"
    c23 = sub(a23, b23, c22)
    a24 = r"ﻘ|ﻕ|ڤ|ﻖ|ﻗ"
    b24 = r"ق"
    c24 = sub(a24, b24, c23)
    a25 = r"ڭ|ﻚ|ﮎ|ﻜ|ﮏ|ګ|ﻛ|ﮑ|ﮐ|ڪ|ك|ﻙ"
    b25 = r"ک"
    c25 = sub(a25, b25, c24)
    a26 = r"ﮚ|ﮒ|ﮓ|ﮕ|ﮔ"
    b26 = r"گ"
    c26 = sub(a26, b26, c25)
    a27 = r"ﻟ|ﻝ|ﻞ|ﻠ|ڵ"
    b27 = r"ل"
    c27 = sub(a27, b27, c26)
    a28 = r"ﻡ|ﻤ|ﻢ|ﻣ"
    b28 = r"م"
    c28 = sub(a28, b28, c27)
    a29 = r"ڼ|ﻦ|ﻥ|ﻨ"
    b29 = r"ن"
    c29 = sub(a29, b29, c28)
    a30 = r"ۆ|ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ|ٶ|ﺅ"
    b30 = r"و"
    c30 = sub(a30, b30, c29)
    a31 = r"ﺔ|ﻬ|ھ|ﻩ|ﻫ|ﻪ|ۀ|ە|ة|ہ|ۃ"
    b31 = r"ه"
    c31 = sub(a31, b31, c30)
    a32 = r"ئ|ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|ي"
    b32 = r"ی"
    c32 = sub(a32, b32, c31)
    a33 = r'¬'
    b33 = r'‌'
    c33 = sub(a33, b33, c32)
    # pa0 = r'•|·|●|·|・|∙|｡|ⴰ'
    # pb0 = r'.'
    # pc0 = sub(pa0, pb0, c33)
    pa1 = r',|٬|٫|‚|，'
    pb1 = r'،'
    pc1 = sub(pa1, pb1, c33)
    pa2 = r'ʕ'
    pb2 = r'؟'
    pc2 = sub(pa2, pb2, pc1)
    na0 = r'۰|٠'
    nb0 = r'0'
    nc0 = sub(na0, nb0, pc2)
    na1 = r'۱|١'
    nb1 = r'1'
    nc1 = sub(na1, nb1, nc0)
    na2 = r'۲|٢'
    nb2 = r'2'
    nc2 = sub(na2, nb2, nc1)
    na3 = r'۳|٣'
    nb3 = r'3'
    nc3 = sub(na3, nb3, nc2)
    na4 = r'۴|٤'
    nb4 = r'4'
    nc4 = sub(na4, nb4, nc3)
    na5 = r'۵'
    nb5 = r'5'
    nc5 = sub(na5, nb5, nc4)
    na6 = r'۶|٦'
    nb6 = r'6'
    nc6 = sub(na6, nb6, nc5)
    na7 = r'۷|٧'
    nb7 = r'7'
    nc7 = sub(na7, nb7, nc6)
    na8 = r'۸|٨'
    nb8 = r'8'
    nc8 = sub(na8, nb8, nc7)
    na9 = r'۹|٩'
    nb9 = r'9'
    nc9 = sub(na9, nb9, nc8)
    ea1 = r'ـ|ِ|ُ|َ|ٍ|ٌ|ً|'
    eb1 = r''
    ec1 = sub(ea1, eb1, nc9)
    Sa1 = r'( )+'
    Sb1 = r' '
    Sc1 = sub(Sa1, Sb1, ec1)
    Sa2 = r'(\n)+'
    Sb2 = r'\n'
    Sc2 = sub(Sa2, Sb2, Sc1)
    Sa3 = r'ﻻ|ﻼ'
    Sb3 = r'لا'
    Sc3 = sub(Sa3, Sb3, Sc2)
    Sa4 = r'ﷲ'
    Sb4 = r'الله'
    Sc4 = sub(Sa4, Sb4, Sc3)
    return sub(zero_space, '', Sc4)

# print(sub_alphabets('ئ|ﭛ|ﻯ|ۍ|ﻰ|ﻱ|ﻲ|ں|ﻳ|ﻴ|ﯼ|ې|ﯽ|ﯾ|ﯿ|ێ|ے|ى|يۆ|ވ|ﯙ|ۈ|ۋ|ﺆ|ۊ|ۇ|ۏ|ۅ|ۉ|ﻭ|ﻮ|ؤ'))

#
# with open('char.txt', 'r', encoding='utf8') as c:
#     for char in c:
#         if char != '\n':
#             print(char, '        ', normalizer(char))
# print(sub_alphabets('ٶ'))

