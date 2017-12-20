# Command on Seth's machine from this directory is
# ./pro_game_analysis.sh -a -g 100 -w ../../weights.txt ../../../training/games/ ../../autogtp/

function print_help {
    echo "
usage: $0 [-h] [-a] [-b] [-w weights_txt_file ] [-g games] game_dir network_dir

This script analyzes leelaz performance with all the networks under network_dir
to give a rough sense of the quality of each network.

optional arguments:
  -h              show this help message and exit
  -a              Run analysis (verus just building files)
  -b              Build leelaz in the build folder
  -w weights      Path of supervised weights.txt
  -g games        Number of games to test over Default: 1000

Positional arguments
  game_dir     Directory of professions sgf files
  network_dir  Directory containing network files to test
"
}

GAMES=1000
TMP_DIR=../../pro_game_build
while getopts habw:g: option
do
 case "${option}"
 in
 h) print_help; exit 2;;
 a) ANALYSIS=1;;
 b) BUILD=1;;
 w) WEIGHT_FILE=$OPTARG;;
 g) GAMES=$OPTARG;;
 esac
done

# remove parsed arguements
shift $((OPTIND-1))

# Arg checking
if [ "$#" -ne 2 ]; then
    print_help;
    exit 2;
fi

GAME_DIR=$1
NETWORK_DIR=$2

if [ ! -d "$1" -o ! -d "$2" ]; then
    echo "game_dir(\"$1\") or network_dir(\"$2\") does not exist"
    exit 2;
fi

# Build
if [ ! -z "$BUILD" ]; then
    mkdir -p "$TMP_DIR"
    cd pro_game_build
    cmake ..
    make
    cd -
fi

# Collect games
echo "collecting $GAMES games from $GAME_DIR for testing"
GAMES_TO_TEST=$TMP_DIR/game_list.txt
SGF_TO_TEST=$TMP_DIR/games.sgf
find "$GAME_DIR" -iname "*.sgf" | sort -R | head -n$GAMES > "$GAMES_TO_TEST"
echo "collected `wc -l < "$GAMES_TO_TEST"` games into $GAMES_TO_TEST"
cat "$GAMES_TO_TEST" | xargs cat > "$SGF_TO_TEST"

# Analysis
if [ ! -z "$ANALYSIS" ]; then
    testNetwork() {
        net=$1
        file_date=`stat -c "%y" "$net"`
        net_name=`basename "$net"`
        results=`echo "test_supervised $SGF_TO_TEST" | "$TMP_DIR/leelaz" -q -s 123 -w $net | tail -n4 | head -n1 |
            sed 's#[^,]*[/ ]\([a-f0-9.%]*\)\(,\|$\)#\1, #g'`
        echo "$file_date, $net_name, $results"
    }


    echo
    find "$NETWORK_DIR" -regextype sed -regex ".*/[a-f0-9]\{64\}" | xargs ls -tr | while read net; do
        testNetwork "$net"
    done

    if [ ! -z "$WEIGHT_FILE" ]; then
        testNetwork "$WEIGHT_FILE"
    fi
fi
