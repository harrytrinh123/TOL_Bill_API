# bill_detect (OCR bill detection API)
Bill detection and storage, visualize data
## Function
- Detect return image in base64
- Detect return data in json
- Storage data
- Visualize data
## Tools
- YOLO v4
- pytesseract
- Flask
- base64
# ============LOCAL==================
## Install
- $ pip install requirements.txt
- Installing tesseract on Windows is easy with the precompiled binaries found https://digi.bib.uni-mannheim.de/tesseract/.
- Copy file vie.traineddata to C:\Program Files\Tesseract-OCR\tessdata
- Run app.py
## Storage
- Database(mongodb)
- Storage image (cloudinary)
## database:
- Bill
  - id
  - datetime
  - total
  - address
  - detail
  - imgUrl
- itemId
  - Item
  - id
  - content

# ==========DEPLOYMENT==============
## Install package
- virtualenv env
- Activate env: ./env/Scripts/activate
- If can not activate env:
  - Open cmd as an Administrator
  - Set-ExecutionPolicy Unrestricted -Force
- Deactivate: deactivate
- pip install -r requirements.txt
- pip freeze > requirements.txt
- Notes:
  - module opencv-python not work on heroku
  - add opencv-contrib-python-headless to requirements.txt
  - heroku create opencvcheck2


## Install tesseract-orc on heroku
- Add apt buildpack to heroku
  - heroku buildpacks:add --index 1 https://github.com/heroku/heroku-buildpack-apt
- Create a file named Aptfile in the same directory as your app and these lines to it:
  - tesseract-ocr
  - tesseract-ocr-vie
  - imagemagick
-Set the config variable using
  - heroku config:set TESSDATA_PREFIX=/app/.apt/usr/share/tesseract-ocr/4.00/tessdata
- heroku restart

## Deploy application
$ heroku login

$ cd tolbill/
$ git init
$ heroku git:remote -a tolbilltest

$ git add .
$ git commit -am "make it better"
$ git push heroku master
