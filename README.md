**Distributed DL-Image-Localization
**

This is a code base used for applying localization models to different RF datasets, focusing primarily on the Helium Network's Proof-of-Coverage data.

This software employs celery workers to parallelize model training and test across containers on distributed hosts. It can also be run self-contained on a local host using docker containers.

The original CNN models and starting point for this code base are forked from work by Dr. Frost Mitchell, which can be found at https://git-os.flux.utah.edu/frost/dl-image-localization

A large RF dataset was extracted and curated from the Helium Network's blockchain and can be found here https://zenodo.org/records/15478183



