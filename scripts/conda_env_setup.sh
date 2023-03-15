#!/bin/bash

cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/activate.d/env_vars.sh
echo  >> ./etc/conda/activate.d/env_vars.sh
echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib" >> ./etc/conda/activate.d/env_vars.sh

echo '#!/bin/sh' > ./etc/conda/deactivate.d/env_vars.sh
echo >> ./etc/conda/deactivate.d/env_vars.sh
echo "unset LD_LIBRARY_PATH" >> ./etc/conda/deactivate.d/env_vars.sh

echo "Contents of $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh:"
cat ./etc/conda/activate.d/env_vars.sh
echo ""

echo "Contents of $CONDA_PREFIX/etc/conda/deactivate.d/env_vars.sh:"
cat ./etc/conda/deactivate.d/env_vars.sh
echo ""
