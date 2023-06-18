current_dir=$(pwd)

for dd in $(find . -type d -name "test*"); do

  cd $dd

  for fullfile in $(ls *.tar.gz); do
    tar -xvf $fullfile -C .
    rm -rf $fullfile
    sleep 1
  done

  cd big_images

  for fullfile in $(ls *.zip); do

    filename=$(basename -- "$fullfile")
    extension="${filename##*.}"
    prefix="${filename%.*}"

    echo $prefix

    if [ -f $prefix.dzi ]; then
      rm -rf ${prefix}.dzi ${prefix}_files
      sleep 1
    fi

    unzip $fullfile
    mv $prefix/* .
    rm -rf $prefix
    rm -rf $fullfile
  done

  cd $current_dir

done
