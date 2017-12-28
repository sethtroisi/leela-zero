#!/usr/bin/env python3

import argparse
import json
import os.path
import re
import gzip
import operator
import shutil
import urllib.request


NETWORKS_URL="http://zero.sjeng.org/networks/"
NETWORKS_JSON_URL="http://zero-test.sjeng.org/data/elograph.json"


def find_networks():
    #with urllib.request.urlopen(NETWORKS_URL) as response:
    #    html = response.read()
    #    f = open("tmp_url", "wb")
    #    f.write(html)
    #    f.close()
    all_networks = {}

    with open("tmp_url", "r") as f:
        html = f.read()
        # TODO grab date
        rows = re.finditer(r"href=\"([^\"]*\.gz)\".*<td>(2017-[^<]*)</td>", html, re.I)
        for row in rows:
            all_networks[row.group(1)] = {"date": row.group(2)}

    names = sorted(all_networks.keys())
    print("Found {} networks ({} to {})".format(
        len(all_networks), names[0], names[-1]))
    return all_networks


def find_network_info(networks):
    #with urllib.request.urlopen(NETWORKS_JSON_URL) as response:
    #    html = response.read()
    #    f = open("tmp_url2", "wb")
    #    f.write(html)
    #    f.close()

    merged_info = {}

    with open("tmp_url2", "r") as f:
        html = f.read()
        raw = json.loads(html)

        for row in raw:
            prefix_name = row["hash"]
            for name, info in networks.items():
                if name.startswith(prefix_name):
                    merged_info[name] = {**row, **info}
                    break
            else:
                print("Didn't find network for", row)
                continue

    unused_networks = networks.keys() - merged_info.keys()
    if unused_networks:
        print("{} networks without info ({})".format(
            len(unused_networks), ", ".join(sorted(unused_networks))))

    return merged_info


def download_networks(args, info):
    if args.all_networks:
        print("Downloading all networks")
        assert False, "Not Implemented: resource heavy"

    count = 0
    for network, data in info.items():
        if data["best"] == 'true':
            name = network.replace(".gz", "")
            file_path = os.path.join(args.network_dir, name)
            if not os.path.exists(file_path):
                url = NETWORKS_URL + network
                gzip_path = file_path + ".gz"
                print("Downloading \"{}\" to \"{}\"".format(network, gzip_path))
                urllib.request.urlretrieve(url, gzip_path)
                print("Extracting")
                with gzip.GzipFile(gzip_path) as f_in:
                    with open(file_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    print ("Mtime:", f_in.mtime)
                    data["mtime"] = f_in.mtime
                assert os.path.exists(file_path), "{} doesn't exist".format(file_path)
                count += 1

    if count > 0:
        print ("Downloaded and extracted {} networks to {}".format(
            count, args.network_dir))


def store_data(args, info):
    count = 0
    with open(args.output_csv, "w") as f_out:
        f_out.write("hash,date,games,number,rating\n")

        # Sort by games
        for item in sorted(info.items(), key=lambda x: int(x[1]["net"])):
            name, data = item
            if data["best"] == 'true':
                print ("\t", name, data)
                date = data["date"]
                games = int(data["net"])
                count += 1
                rating = data["rating"]
                f_out.write(",".join(map(str, [name, date, games, count, rating])) + "\n")


if __name__ == "__main__":
    usage_str = """
This script does find info on "production" network and downloads networks to a network directory
"""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=usage_str)
    parser.add_argument("-n", "--network_dir", metavar="network_dir", dest="network_dir", type=str,
                        default="", help="directory to store downloaded networks")
    parser.add_argument("-a", "--all_networks", dest="all_networks", type=bool,
                        default=False, help="download all networks (default: production only)")
    parser.add_argument("-o", "--output_csv", dest="output_csv", type=str,
                        default="network.csv", help="csv file to output data to")
    args = parser.parse_args()

    networks = find_networks()
    info = find_network_info(networks)
    download_networks(args, info)
    store_data(args, info)

