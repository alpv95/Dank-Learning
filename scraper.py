from __future__ import division
from bs4 import BeautifulSoup
import requests
import shutil
import os.path

save_path = 'memes'
n_captions = 14   #this number *15 is the total number of captions per template
n_templates = 181  #this number *15 is the total number of templates
Uerrors = 0

for i in range(1,n_templates):
    if i == 1:
        url = 'https://memegenerator.net/memes/popular/alltime'
    else:
        url = 'https://memegenerator.net/memes/popular/alltime/page/' + str(i)

    r = requests.get(url)
    soup = BeautifulSoup(r.text,'html.parser')
    chars = soup.find_all(class_='char-img')
    links = [char.find('a') for char in chars]
    imgs = [char.find('img') for char in chars]
    assert len(links) == len(imgs)
    for j,img in enumerate(imgs):
        img_url = img['src']
        response = requests.get(img_url, stream=True)
        name_of_file = img_url.split('/')[-1]
        completeName = os.path.join(save_path, name_of_file)
        with open(completeName,'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        for k in range(1,n_captions):
            if k == 1:
                URL = 'https://memegenerator.net' + links[j]['href']
            else:
                URL = 'https://memegenerator.net' + links[j]['href'] + '/images/popular/alltime/page/' + str(k)

            R = requests.get(URL)
            SOUP = BeautifulSoup(R.text,'html.parser')
            CHARS = SOUP.find_all(class_='char-img')
            IMGS = [char.find('img') for char in CHARS]
            with open('Captions.txt', 'a') as f:
                for IMG in IMGS:
                    try:
                        f.write('%s\n' % (IMG['alt']))
                    except UnicodeEncodeError:
                        Uerrors += 1
                        pass
    if i % 10 == 0:
        print('<'+'='*10+'>')
        print(i)
        print(Uerrors)



URL = 'https://memegenerator.net' + links[0]['href']
R = requests.get(URL)
SOUP = BeautifulSoup(R.text,'html.parser')
CHARS = SOUP.find_all(class_='char-img')
IMGS = [char.find('img') for char in CHARS]
print(IMGS[1]['alt'])


img_url = imgs[0]['src']
response = requests.get(img_url, stream=True)
name_of_file = img_url.split('/')[-1]
completeName = os.path.join(save_path, name_of_file)
with open(completeName,'wb') as out_file:
    shutil.copyfileobj(response.raw, out_file)
del response



