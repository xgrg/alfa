# How to use DARTEL scripts (`alfa/spm`)

One script `DARTEL.py` is used to generate a DARTEL template from previously
realigned sets of grey maps (`r*_c1.nii`) (NB: running the process in CLI or from
Matlab directly may save the overhead time due to Python/Nipype).
The script takes the path of the directory containing the realigned images and
a JSON file with a list of identifiers (allowing to select/discard subjects)

Then the registration of any modality to the generated template is done in four
steps:

- Co-registration `CoregDWI.py` of the given modality with respect to the T1/c1
(grey) maps.

- Spatial normalization `DARTELNorm2MNI.py` using the corresponding flowfields
and the generated template.

- Removal of the headers `remove_headers.py` from the resulting outputs
`(s)wr_*.nii`. This step is a walkaround to work out potential later confusions
with SPM handling referentials/transformations.

- Final co-registration with the MNI template `CoregMNI.py` (the spatial
  normalization is supposed to include this but the headers get removed). The
  goal is to obtain images that superimpose natively one onto the other without
  relying on their headers.
