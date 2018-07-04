from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import os
#import sys
#sys.path.insert(0, '~/PycharmProjects/MemeProject/im2txt/Jmemes')

class ImageText(object):
    def __init__(self, filename_or_size, desired_size=300, mode='RGBA', background=(0, 0, 0, 0),
                 encoding='utf8'):
        if isinstance(filename_or_size, str):
            self.filename = filename_or_size
            image = Image.open(self.filename)
            baseheight = desired_size
            hpercent = (baseheight/float(image.size[1]))
            wsize = int((float(image.size[0])*float(hpercent)))
            self.image = image.resize((wsize,baseheight), Image.ANTIALIAS)
            self.size = self.image.size
        elif isinstance(filename_or_size, (list, tuple)):
            self.size = filename_or_size
            self.image = Image.new(mode, self.size, color=background)
            self.filename = None
        self.draw = ImageDraw.Draw(self.image)
        self.encoding = encoding

    def save(self, filename=None):
        self.image.save(filename or self.filename)

    def get_font_size(self, text, font, max_width=None, max_height=None):
        if max_width is None and max_height is None:
            raise ValueError('You need to pass max_width or max_height')
        font_size = 1
        text_size = self.get_text_size(font, font_size, text)
        if (max_width is not None and text_size[0] > max_width) or \
           (max_height is not None and text_size[1] > max_height):
            raise ValueError("Text can't be filled in only (%dpx, %dpx)" % \
                    text_size)
        while True:
            if (max_width is not None and text_size[0] >= max_width) or \
               (max_height is not None and text_size[1] >= max_height):
                return font_size - 1
            font_size += 1
            text_size = self.get_text_size(font, font_size, text)

    def write_text(self, (x, y), text, font_filename, font_size=11,
                   color=(0, 0, 0), max_width=None, max_height=None):
        if isinstance(text, str):
            text = text.decode(self.encoding)
        if font_size == 'fill' and \
           (max_width is not None or max_height is not None):
            font_size = self.get_font_size(text, font_filename, max_width,
                                           max_height)
        text_size = self.get_text_size(font_filename, font_size, text)
        font = ImageFont.truetype(font_filename, font_size)
        if x == 'center':
            x = (self.size[0] - text_size[0]) / 2
        #if y == 'center':
            #y = (self.size[1] - text_size[1]) / 2
        self.draw.text((x, y), text, font=font, fill=color)
        return text_size

    def get_text_size(self, font_filename, font_size, text):
        font = ImageFont.truetype(font_filename, font_size)
        return font.getsize(text)

    def write_text_box(self, (x, y), text, box_width, font_filename,
                       font_size=11, color=(0, 0, 0), place='left',
                       justify_last_line=False,bottom=False):
        lines = [1,2,3]
        words = text.split()
        while len(lines) >= 3:
            line=[]
            lines = []
            for word in words:
                new_line = ' '.join(line + [word])
                size = self.get_text_size(font_filename, font_size, new_line)
                text_height = size[1]
                if size[0] <= box_width:
                    line.append(word)
                else:
                    lines.append(line)
                    line = [word]
            if line:
                lines.append(line)
            lines = [' '.join(line) for line in lines if line]
            if len(lines) >= 3:
                font_size -= 3

        if bottom and len(lines) > 1:
            y -= font_size

        height = y
        for index, line in enumerate(lines):
            if index != 0:
                height += text_height
            if place == 'left':
                self.write_text((x, height), line, font_filename, font_size,
                                color)
            elif place == 'right':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = x + box_width - total_size[0]
                self.write_text((x_left, height), line, font_filename,
                                font_size, color)
            elif place == 'center':
                total_size = self.get_text_size(font_filename, font_size, line)
                x_left = int(x + ((box_width - total_size[0]) / 2))
                self.write_text((x_left, height), line, font_filename,
                                font_size, color)
            elif place == 'justify':
                words = line.split()
                if (index == len(lines) - 1 and not justify_last_line) or \
                   len(words) == 1:
                    self.write_text((x, height), line, font_filename, font_size,
                                    color)
                    continue
                line_without_spaces = ''.join(words)
                total_size = self.get_text_size(font_filename, font_size,
                                                line_without_spaces)
                space_width = (box_width - total_size[0]) / (len(words) - 1.0)
                start_x = x
                for word in words[:-1]:
                    self.write_text((start_x, height), word, font_filename,
                                    font_size, color)
                    word_size = self.get_text_size(font_filename, font_size,
                                                    word)
                    start_x += word_size[0] + space_width
                last_word_size = self.get_text_size(font_filename, font_size,
                                                    words[-1])
                last_word_x = x + box_width - last_word_size[0]
                self.write_text((last_word_x, height), words[-1], font_filename,
                                font_size, color)
        return (box_width, height - y)

images = ['evil-raccoon.jpg','annoying-gamer-kid.jpg','i-have-no-idea-what-im-doing-dog-with-tie.jpg' \
          ,'blackjack-and-hookers-bender.jpg','blackjack-and-hookers-bender.jpg','cool-dog.jpg','look-marge.jpg','newspaper-cat-realization.jpg' \
          ,'grumpy-cat-good.jpg','skeptical-third-world-kid.jpg','aand-its-gone.jpg','musically-oblivious-8th-grader.jpg','really-high-guy.jpg','matrix-morpheus.jpg', 'sudden-realization-ralph.jpg' \
          , 'business-cat.jpg','the-most-interesting-man-in-the-world.jpg','yo-dawg.jpg']

images_TA = ['kian.jpg','surag.jpg','akash.jpg','patrick.jpg','yoann.jpg','suraj.jpg','guillaume.jpg','olivier.jpg','ramtin.jpg','andrew.jpg','akash.jpg','guillaume.jpg','aarti.jpg' \
          , 'xin.jpg','surag.jpg']

TEXT1_TA = ["i don't", "oh , you're an expert?","this is what happens", "i don't always miss penalties", "are you the only one", "went to bed with a hot girl", "hi , my name is inigo montoya" \
, "i don't know how to use the elevator","the face you make when", "and then i said...","i don't always cry"
    , "my face when i see a hot chick", "my name is luke skywalker"," this is how i feel when","the moment when you realize that"]

TEXT1=['yes , yes , yes',
'why do i have to wait',
'have you ever heard of these memes?',
"i'll make my own eurovision",
 "i'll start my own church",
"hey i just met you , and this is crazy",
"look marge",
 'i should start studying for finals',
"i made a meme once",
 "so you're telling me",
"hope you have a happy birthday",
 'that awkward moment when',
'dude',
 "what if i told you",
"the moment you realize",
"hi,",
"i don't always get high but when i do,",
"yo dawg, I heard you like memes",
]

TEXT2_TA = ["want to be a stripper anymore", "tell me more","when you go to sleep", "but when i do...", "who does not want to be in the kitchen", " ", "and this is jackass" \
, " ","you realize your girlfriend is watching you", "i told you!!!","but when i do , it's in the morning"
    , " ", "and i love lord of the rings"," i look at my teeth","you don't have enough money"]

TEXT2=[' ',
'until i get a job',
' ',
"with blackjack and hookers!",
 "with blackjack and hookers",
"but i'm a dog",
"i'm a meme",
 ' ',
"it was awful.",
 "that you don't have enough money to spend money on the internet?",
"aaand it's gone",
 'you get out of the friendzone with your mom',
'i have no nose',
 "that you don't have to go to the gym without posting on facebook",
"that you don't have to be an adventurer",
"i'm a cat",
"i use a distribution",
"so i took an arrow to the knee"]

images_TA = ['im2txt/Jmemes/IMG_8140.jpg','im2txt/Jmemes/IMG_8174.jpg']
TEXT1_TA = ["look at me","this guy is my dog"]
TEXT2_TA = ["i'm a ginger"," "]

for i,ting in enumerate(images_TA):
    color = (255, 255, 255)
    text1 = TEXT1_TA[i].upper()
    text2 = TEXT2_TA[i].upper()
    font = 'impact.ttf'
    img = ImageText(images_TA[i]) #background=(255, 255, 255, 200)) # 200 = alpha

    #write_text_box will split the text in many lines, based on box_width
    #`place` can be 'left' (default), 'right', 'center' or 'justify'
    #write_text_box will return (box_width, box_calculed_height) so you can
    #know the size of the wrote text
    #img.write_text_box((0, 0), text, box_width=100, font_filename=font,
                       #font_size=15, color=color)
    #img.write_text_box((50, 50), text, box_width=200, font_filename=font,
                       #font_size=15, color=color, place='right')
    img.write_text_box((2, 0), text1, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center')
    img.write_text_box((-2, 0), text1, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center')
    img.write_text_box((0, -2), text1, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center')
    img.write_text_box((0, 2), text1, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center')
    img.write_text_box((0, 0), text1, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=color, place='center')

    img.write_text_box((-2, 255), text2, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center',bottom=True)
    img.write_text_box((2, 255), text2, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center',bottom=True)
    img.write_text_box((0, 253), text2, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center',bottom=True)
    img.write_text_box((0, 257), text2, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=(0,0,0), place='center',bottom=True)
    img.write_text_box((0, 255), text2, box_width=img.size[0]+2, font_filename=font,
                       font_size=32, color=color, place='center',bottom=True)

    #You don't need to specify text size: can specify max_width or max_height
    # and tell write_text to fill the text in this space, so it'll compute font
    # size automatically
    #write_text will return (width, height) of the wrote text
    #img.write_text((0, 50), 'TEST FILL', font_filename=font,
                   #font_size=40, max_height=150, color=(0,0,0))
    #img.write_text((2, 2), 'TEST FILL LA LA AL AL ALLA LA LA LALALAL LALA LALA LALAL L', font_filename=font,
                   #font_size=31, max_height=150, color=color)
    try:
        img.save(text2.split()[-1]+ str(i) + '.jpg')
    except:
        img.save(text1.split()[-1] + '.jpg')
