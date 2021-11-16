def DankGrammar(sentence):
    #given a input list of strings, joins them in a grammatically correct fashion
    #outputs a single string, the final joined sentence
    final = sentence[0]
    for i,word in enumerate(sentence[1:]):
        if final[-1] != ',' and final[-1] != '.' and final[-1] != '!' and final[-1] != '?' and final[-1] != '*' and final[-1] != "'" and final[-1] != ':' and final[-1] != ';' and final != '%' and final != '...' and final != '$':
            if word == ',' or word == '.' or word == '!' or word == '?' or word == '*' or word == "'" or word == ':' or word == ';' or word == '%' or word == '...':
                final += word
            else:
                final += ' ' + word
        else:
            if final[-1] == ',' or final[-1] == '.' or final[-1]==':' or final[-1]==';' or final[-1] == '%' or final[-1] == '...':
                if word == '.':
                    final += word
                else:
                    final += ' ' + word
            elif final[-1] == '!' or final[-1] == '?':
                if word == '!' or word == '?' or word == '*':
                    final += word
                else:
                    final += ' ' + word
            else:
                final += word

    return final

def DankStrip(sentence):
    #takes list of strings, splits the sentence, return two lists of strings, upper and lower text

    #with punctuation:
    punct = []
    deleters = []
    for i,word in enumerate(sentence):
        if word == ',' or word == '.' or word == '!' or word == '?' or word == ':' or word == ';' or word == '...':
            punct.append(i)

    if punct:
        for i, punc in enumerate(punct):
            if i != len(punct)-1:
                if punct[i+1] == punc+1:
                    deleters.append(i)

        for i, ting in enumerate(deleters):
            del punct[ting-i]
        print(punct)

        if min(punct) < len(sentence)*4/5:
            upper = sentence[:min(punct)+1]
            lower = sentence[min(punct)+1:]
        else:
            #some special cases, starter words:
            if sentence[0] == 'dude' or sentence[0] == 'hey':
                upper = [sentence[0]]
                lower = sentence[1:]
            elif 'still' in sentence:
                upper = sentence[:sentence.index('still')]
                lower = sentence[sentence.index('still'):]
            else:
                if len(sentence) < 7:
                    upper = sentence
                    lower =[]
                else:
                    #last resort is just to split in half
                    upper = sentence[:int(len(sentence)/2)]
                    lower = sentence[int(len(sentence)/2):]

    #no punctuation
    else:
        #some special cases, starter words:
        if sentence[0] == 'dude' or sentence[0] == 'hey':
            upper = [sentence[0]]
            lower = sentence[1:]
        elif 'still' in sentence:
            upper = sentence[:sentence.index('still')]
            lower = sentence[sentence.index('still'):]
        else:
            if len(sentence) < 6:
                upper = sentence
                lower =[]
            else:
                #last resort is just to split in half
                upper = sentence[:int(len(sentence)/2)]
                lower = sentence[int(len(sentence)/2):]

    return upper,lower


