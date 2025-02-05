# HM-PBLV
## Model Faster RCNN

### Funktionalität des Faster‑R‑CNN
	•	Zweistufiges Detektionsverfahren:
	•	Erste Stufe (Region Proposal Network – RPN):
Das RPN arbeitet auf den gemeinsamen Faltungsmerkmalen (Convolutional Features) des gesamten Bildes und generiert für jede räumliche Position mehrere vordefinierte Anker (Anchor Boxes) unterschiedlicher Größen und Seitenverhältnisse. Dadurch wird eine Menge von potenziellen Objektregionen (Region Proposals) vorgeschlagen, ohne dass zusätzliche, rechenintensive externe Algorithmen wie Selective Search erforderlich sind.
	•	Zweite Stufe (Detektion und Klassifikation):
	•	Für jede vorgeschlagene Region werden die Merkmale extrahiert (über einen Faltungs- und anschließenden Flatten‑Layer) und an zwei separate „Köpfe“ weitergeleitet:
	•	Klassifikationskopf: Liefert mit einer Softmax-Aktivierung Wahrscheinlichkeiten für die zu erkennenden Klassen (hier als Dummy-Vektor mit 50 Werten implementiert, was 10 Objekten mit je 5 Klassen entspricht).
	•	Regressionskopf: Erzeugt lineare Vorhersagen für die Koordinaten der Begrenzungsrahmen (10 Boxen à 4 Koordinaten, also insgesamt 40 Werte).

### Vorteile des Faster‑R‑CNN-Ansatzes
	•	Hohe Genauigkeit:
Durch die zweistufige Architektur werden Region Proposals gezielt generiert und anschließend detailliert klassifiziert und lokalisiert. Diese Struktur ermöglicht eine präzise Objekterkennung – ein wesentlicher Vorteil im Vergleich zu Ein-Schritt-Detektoren, die oft Geschwindigkeit über Genauigkeit stellen.
	•	Effizienz durch Feature-Sharing:
Die gemeinsame Nutzung der Faltungsmerkmale zwischen dem RPN und dem Detektionskopf reduziert den Rechenaufwand erheblich, da das Bild nur einmal durch das vortrainierte Netzwerk (hier ResNet50) geleitet wird.
	•	Flexibilität bei Transfer Learning:
Da als Backbone ein vortrainiertes ResNet50 verwendet wird, profitieren wir von bereits erlernten Merkmalen (zum Beispiel aus ImageNet). Dies führt zu einer schnelleren Konvergenz und besseren Generalisierungsfähigkeit, was insbesondere bei komplexen Datensätzen wie KITTI von Vorteil ist.
￼
### Warum Faster‑R‑CNN für den KITTI-Datensatz?
	•	Realitätsnahe Szenarien:
Der KITTI-Datensatz enthält Bilder aus der autonomen Fahrzeugtechnik – komplexe Umgebungen mit Fahrzeugen, Fußgängern und anderen Verkehrsteilnehmern. Die Fähigkeit von Faster‑R‑CNN, auch kleine und teilweise verdeckte Objekte präzise zu erkennen, ist hier besonders wertvoll.
	•	Robuste Leistung:
Die exzellente Genauigkeit und die Möglichkeit, Transfer-Learning zu nutzen, machen Faster‑R‑CNN zu einer guten Wahl, wenn es darauf ankommt, selbst in herausfordernden Verkehrsszenarien zuverlässige Detektionen zu erzielen.

### Implementierungsüberblick

Die wesentlichen Schritte der Implementierung sind:
	1.	Datenvorverarbeitung:
	•	Der KITTI-Datensatz wird über TensorFlow Datasets (tfds) geladen.
	•	Die Bilder werden auf eine feste Größe (128×128 Pixel) skaliert und normalisiert.
	•	Dummy-Labels werden erzeugt: Ein Dummy-One-Hot-Vektor für die Klassifikation (50 Werte) und ein konstanter Vektor für die Bounding-Box-Vorhersage (40 Werte).
	2.	Modellaufbau:
	•	Backbone: Ein ResNet50-Modell, vortrainiert auf ImageNet, wird als Feature-Extraktor verwendet und dabei eingefroren, um die vortrainierten Merkmale zu erhalten.
	•	Feature-Aufbereitung: Die Merkmale aus dem Backbone werden mittels eines Flatten-Layers in einen eindimensionalen Vektor überführt.
	•	Zwei Ausgabeköpfe:
	•	Der Klassifikationskopf besteht aus einem Dense-Layer, dessen Ausgabe dann umgeformt und mit Softmax normalisiert wird.
	•	Der Regressionskopf besteht aus einem weiteren Dense-Layer, der lineare Vorhersagen für die Bounding Boxes liefert.
	3.	Training und Evaluation:
	•	Der Code definiert eine Trainingsmethode, die das Modell über mehrere Epochen trainiert. Dabei werden TensorBoard-Logs und ein benutzerdefinierter Callback (ProgressCallback) verwendet, um den Fortschritt zu überwachen.
	•	Nach dem Training wird das Modell evaluiert und gespeichert.

Der Fokus der Implementierung liegt auf den Kernelementen – der Datenvorverarbeitung, dem Aufbau des Feature-Extraktors und den zwei separaten Köpfen – ohne sich in detaillierte Code-spezifische Aspekte zu verlieren.
