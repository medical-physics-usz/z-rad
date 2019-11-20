different methods to calculate texture features

Methods for GLCM, GLRLM:
2D:
1) (no suffix) features are computed from each 2D directional matrix and averaged over 2D directions and slices
	---- by slice, without merging
2) (_mbs) features are computed from a single matrix after merging 2D directional matrices per slice,
	and then averaged over slices. 
	---- by slice, with merging by slice:
3) (_mbd) features are computed from a single matrix after merging 2D directional matrices per direction and
	then averaged over directions.
	---- by slice, with merging by direction
4) (_mf) the feature is computed from a single matrix after merging all 2D directional matrices
        ---- by slice, with full merging
-----------------------------------------------------------------
3D:
5) features are computed from each 3D directional matrix and averaged over the 3D directions
6) the feature is computed from a single matrix after merging all 3D directional matrices


for GLSZM, GLDZM, NGTDM:
2D:
1. (no suffix) Features are computed from 2D matrices and averaged over slices (8QNN).
2. (_m) The feature is computed from a single matrix after merging all 2D matrices (62GR).
-----------------------------------------------------------------
3D:
3. The feature is computed from a 3D matrix (KOBO).

