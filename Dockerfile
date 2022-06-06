FROM tensorflow/tensorflow:2.2.2-py3
RUN apt-get update && apt-get install -y git unzip wget curl
RUN pip3 install --upgrade pip
RUN pip3 install cmake --upgrade
WORKDIR scDrug
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .

## scMatch
RUN git clone https://github.com/asrhou/scMatch.git /opt/scMatch

RUN unzip /opt/scMatch/refDB/FANTOM5/10090_HID.csv.zip -d /opt/scMatch/refDB/FANTOM5/ && rm /opt/scMatch/refDB/FANTOM5/10090_HID.csv.zip
RUN unzip /opt/scMatch/refDB/FANTOM5/10090_symbol.csv.zip -d /opt/scMatch/refDB/FANTOM5/ && rm /opt/scMatch/refDB/FANTOM5/10090_symbol.csv.zip
RUN unzip /opt/scMatch/refDB/FANTOM5/9606_HID.csv.zip -d /opt/scMatch/refDB/FANTOM5/ && rm /opt/scMatch/refDB/FANTOM5/9606_HID.csv.zip
RUN unzip /opt/scMatch/refDB/FANTOM5/9606_symbol.csv.zip -d /opt/scMatch/refDB/FANTOM5/ && rm /opt/scMatch/refDB/FANTOM5/9606_symbol.csv.zip 

RUN sed -i 's/\.ix/.loc/g' /opt/scMatch/scMatch.py
RUN sed -i 's/loc\[commonRows, ].fillna(0\.0)/reindex(commonRows, axis="index", fill_value=0.0)/g' /opt/scMatch/scMatch.py

## survival analysis
RUN wget https://figshare.com/ndownloader/files/35596748 -O data/TCGA.zip
RUN unzip data/TCGA.zip
RUN rm data/TCGA.zip

## CaDRReS-Sc
RUN git clone https://github.com/CSB5/CaDRReS-Sc.git /opt/CaDRReS-Sc
RUN wget https://www.dropbox.com/s/3v576mspw5yewbm/GDSC_exp.tsv -O /opt/CaDRReS-Sc/data/GDSC/GDSC_exp.tsv
RUN mkdir -p /opt/CaDRReS-Sc/data/CCLE
RUN wget https://ndownloader.figshare.com/files/29124747 -O /opt/CaDRReS-Sc/data/CCLE/CCLE_expression.csv
RUN mkdir -p /opt/CaDRReS-Sc/preprocessed_data/PRISM
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BdW-3oBp8q5hTTjB3iNysqSf8GNkKGH6' -O /opt/CaDRReS-Sc/preprocessed_data/PRISM/PRISM_drug_info.csv
RUN wget --no-check-certificate 'https://docs.google.com/uc?export=download&id=1PGdLJonMBldTYihruXug-LgyoU7PQpAn' -O /opt/CaDRReS-Sc/preprocessed_data/PRISM/feature_genes.txt

RUN sed -i 's/import tensorflow as tf/import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()/g' /opt/CaDRReS-Sc/cadrres_sc/model.py
RUN sed -i 's/import tensorflow\.python\.util\.deprecation as deprecation/from tensorflow.python.util import deprecation/g' /opt/CaDRReS-Sc/cadrres_sc/model.py

## CIBERSORTx
RUN curl -fsSL https://get.docker.com -o get-docker.sh
RUN sh get-docker.sh


CMD [ "/bin/bash" ]


