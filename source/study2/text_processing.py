import re

def swap_gender(gender):
    if gender == 'F':
        return 'M'
    else:
        return 'F'

def clean(text, stop_words=True):
	stopwords = ['is', 'was', 'are', 'were', 'on', 'in', 'up', 'by', 'or', 'and', 'the', 'a', 'an',
	             'at', 'which', 'when', 'after', 'with', 'as']
	text = text.replace("\n", " ")
	text = text.replace("*", " ")
	text = text.replace("[", " ")
	text = text.replace("]", " ")
	text = text.replace("(", " ")
	text = text.replace(")", " ")
	text = text.replace("-", " ")
	text = text.replace(":", " ")
	text = text.replace("%", " ")
	text = text.replace(".", " ")
	text = text.replace(",", " ")
	text = text.replace(";", " ")
	text = text.replace("#", " ")
	text = text.replace("/", " ")
	text = text.replace("@", " ")
	text = text.replace('?', ' ')
	text = text.replace("{", " ")
	text = text.replace("}", " ")
	text = ''.join([i for i in text if not i.isdigit()])
	# if stop_words:
	#   for word in stopwords:
	#      text = text.replace('\b' + word + '\b', " ")
	#     text = text.replace(' ' + word + ' ', " ")
	text = text.replace("  ", " ")
	return text.lower()


def gender_swapping(text, gender, neutralize=False):

    if neutralize:
        text = re.sub(r"\bher\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+F\b", "Sex: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bshe\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bwoman\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfemale\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bF\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhis\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "its", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+M\b", "Sex: ", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bman\b", "patient", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmale\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\bM\b", "", text, flags=re.IGNORECASE)

    if gender == 'F':
        text = re.sub(r"\bher\b", "his", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+F\b", "Sex: M", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bshe\b", "he", text,  flags=re.IGNORECASE)
        text = re.sub(r"\bwoman\b", "man", text, flags=re.IGNORECASE)
        text = re.sub(r"\bfemale\b", "male", text, flags=re.IGNORECASE)
        text = re.sub(r"\bF\b", "M", text, flags=re.IGNORECASE)

    else:
        text = re.sub(r"\bhis\b", "her", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhim\b", "her", text, flags=re.IGNORECASE)
        text = re.sub(r"\bSex:\s+M\b", "Sex: F", text, flags=re.IGNORECASE)
        text = re.sub(r"\bhe\b", "she", text, flags=re.IGNORECASE)
        text = re.sub(r"\bman\b", "woman", text, flags=re.IGNORECASE)
        text = re.sub(r"\bmale\b", "female", text, flags=re.IGNORECASE)
        text = re.sub(r"\bM\b", "F", text, flags=re.IGNORECASE)

    return text
