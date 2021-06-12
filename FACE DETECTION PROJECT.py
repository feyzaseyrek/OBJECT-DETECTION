
# author : feyzaseyrek
# date: 20.01.2021

import cv2
import imageio

# CASCADE XML İÇİNDE HAZIR EĞİTİLMİŞ MODELLER VAR (OPENCV SAĞLIYOR.)
face_cascade= cv2.CascadeClassifier('haarcascade-frontalface-default.xml')  #YÜZ TANIM İÇİN OPENCV HAARCASCADE MODELİ

eye_cascade= cv2.CascadeClassifier('haarcascade-eye.xml')  # GÖZ TANIMA İÇİN OPENCV HAARCASCADE MODELİ

def detect(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #CASCADE ler siyah-beyaz resim üzerinde nesne tanıdığı için resmi önce siyah beyaza çevirdik.
    faces= face_cascade.detectMultiScale(gray, 1.3, 5) #resimde yüz tanımak için cascadelerin sağladığı detectMultiScale fonksiyonu (en az 5 pencere arıyor)
    for(x, y, w, h) in faces: #4 elemanlı tuple döndürüyor. x ve y çerçevenin koordinarlarını, w ve h çerçevenin en boyunu ayarlar.
        cv2.rectangle(frame, (x,y), (x+w,y+h), (255, 0, 0), 2) #frame=renkli resim , (x+w,y+h) sağ alt köşenin koordiantları , Rgb (255=kırmızı) , 2 piksel kalınlık
        gray_face = gray[y:y+h, x:x+w]  # parametre olarak yüzü çerçeveleyen dikdörtgenin koordinatlatını alıyoruz.
        color_face = frame[y:y+h, x:x+w]
        eyes= eye_cascade.detectMultiScale(gray_face, 1.1, 3) #resimdeki gözleri tespit eder.
        for(ex, ey , ew, eh) in eyes:
            cv2.rectangle(color_face, (ex,ey), (ex+ew,ey+eh), (0, 255, 0), 2) # RGB (0,255,0)= YEŞİL , 2 Piksel kalınlık
    return frame

reader = imageio.get_reader('input.mp4') #input olan videoyu okuyor.
fps = reader.get_meta_data()['fps'] #videoda kaç fps taradığını bastırıyor.
writer = imageio.get_writer('outputt.mp4', fps=fps) #yüz tespiti yapılmış video yazılıyor.
for i, frame in enumerate(reader):
    frame= detect(frame)
    writer.append_data(frame)  #videoda kaç fps taradığını bastırıyor.
    print(i)

writer.close()


