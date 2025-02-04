# **Anforderungsanalyse: Objekterkennung von Fahrzeugen in Bildern**

_Gruppe 1: Dino Dervisevic, Leon Bender, Frederic Kayser, Isabella Schmid, Simon Gaertner_

## **1. Projektbeschreibung**

Das Ziel dieses Projekts ist die Entwicklung eines Deep-Learning-Modells zur Erkennung von Fahrzeugen in Bildern. Die
Erkennung erfolgt in zwei Dimensionen (2D) mit Bounding Boxes. Das Modell soll auf dem KITTI-Dataset trainiert werden.

## **2. Funktionale Anforderungen**

### **2.1. Datengrundlage**

- Nutzung des **KITTI-Datasets** für Training und Validierung.
- Laden und Vorverarbeitung der Daten für 2D-Annotationen.

### **2.2. Modelltraining**

- Entwicklung von zwei eigenen neuronalen Netzes mit **TensorFlow**.
- Training des **2D-Modells** zur Objekterkennung mittels Bounding Boxes.
- Zusätzlich soll das Modell eine Klassifikation und Multi-Objekt-Erkennung ermöglichen.
- Optional ist die Erweiterung des Modells um **3D-Tiefenschätzung** durch zusätzliche Netzwerkschichten oder ein
  separates Modell.
- Verwendung einer geeigneten **Loss-Funktion** für Objekterkennung.
- **Ensemble Training**: Kombination mehrerer Modelle zur Verbesserung der Erkennungsgenauigkeit und Robustheit.

### **2.3. Modellbewertung**

- Verwendung geeigneter Metriken wie **IoU (Intersection over Union)** für 2D.
- Erstellung von **Visualisierungen** zur Evaluation der Modellleistung. Beispiele: Visualisierung der
  Loss-Kurven, IoU-Verteilung und Confusion Matrix

### **2.4. Implementierung und Laufzeitumgebung**

- Verwendung von **Python** mit TensorFlow als Framework.
- Bereitstellung einer **REST-API** für die Anwendung des Modells auf neue Bilder und das Anstoßen von
  Trainingsläufen.
- Verwendung von FastAPI als Framework für die REST-API.
- Speicherung von Modellen zur **Wiederverwendung** und **Weiterentwicklung**.

## **3. Nicht-funktionale Anforderungen**

- **Performance**: Modell soll eine performante Verarbeitung ermöglichen.
- **Skalierbarkeit**: Möglichkeit zur Erweiterung auf größere Datensätze oder komplexere Modelle.
- **Modularität**: Trennung von Datenverarbeitung, Modelltraining und Evaluierung für einfache Wartung und Erweiterung.
- **Kompatibilität**: Anwendung soll auf einem Linux System lauffähig sein.

## **4. Herausforderungen und Risiken**

- **Datenqualität**: KITTI enthält unterschiedliche Lichtverhältnisse und Wetterbedingungen, die die Modellgenauigkeit
  beeinflussen können.
- Optionale **3D-Tiefenschätzung**: Erfordert eine geeignete Architektur
- **Rechenaufwand**: Training eines vollständig neuen Modells ist rechenintensiv
- **Ensemble Training**: Erfordert zusätzliche Rechenressourcen und eine geeignete Methode zur Kombination der Modelle.

## **5. Daraus resultierende Aufgaben**

- **Datenbeschaffung**: Herunterladen und Vorverarbeiten des KITTI-Datasets.
- **Modellentwicklung**: Implementierung und Training der Modelle.
- **Ensemble Training**: Zusammenführung und Training mehrerer Modelle.
- **Evaluation**: Bewertung der Modellleistung und Anpassung der Architektur.
- **Implementierung**: Erstellung der REST-API und Speicherung von Modellen.
- **Dokumentation**: Verwendung von Markdown als Dokumentationsformat.