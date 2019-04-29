# Fotoalbum Extractor

## Beschreibung

Die hier entwickelte Software extrahiert aus einer eingescannten Fotoalbumseite die darin enthaltenen Fotos. Dazu sollte die Seite des Fotoalbums sauber eingescannt sein mit einem möglichst gleichfarbigen Hintergrund und ohne Spiegelungen. Die Fotos können einen Rahmen haben, der von dem Programm entfernt wird.
Sobald die Fotos ausgeschnittenen wurden kann für jedes einzelne Foto innerhalb der Fotoalbumseite eine Gesichtserkennung durchgeführt werden, bei der die erkannten Gesichter markiert und ausgeschnitten werden.
Um das Ergebnis der Software zu überprüfen soll es möglich sein die ausgeschnittenen Fotos mit Ground Truth Bildern auf Ähnlichkeit zu vergleichen. 

## Setup für die Verwendung

Das Projekt läuft mit:
* Python 3.6.5 
* numpy 1.14.2
* opencv-python 3.4.0

Nach der Installation von Python kann man mit folgendem Befehl die Abhängigkeiten installieren:

```
pip install -r requirements.txt
```

Das Projekt kann man dann mit folgendem Befehl ausführen sobald man in das Installtionsverzeichniss gewechselthat:

```
python main.py /path/to/image.tif
```

Eine Hilfe zur Ausführung des Progamms lässt sich wie folgt aufrufen:
```
python main.py -h
```