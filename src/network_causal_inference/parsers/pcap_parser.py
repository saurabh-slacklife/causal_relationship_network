from scapy.all import *
from scapy.all import Ether, ARP, srp, send
import pandas as pd
from pandas import DataFrame
import logging
logger = logging.getLogger(__name__)


def load_data(path='../../../data/train/',encoding='utf8') -> DataFrame:
    input_csv_files = glob.glob(os.path.join(path, "*.csv"))
    combined_df_list=list()
    for csv_file in input_csv_files:
        df = pd.read_csv(csv_file,encoding=encoding)
        combined_df_list.append(df)
    return pd.concat(combined_df_list)

def read_pcap_file(pcap_file_path: str, max_packet_to_read: int=10):
    for packet in PcapReader(pcap_file_path):
        try:
            print(packet.summary())
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            raise
        except Exception:
            raise


if __name__ == '__main__':
    path = '../../../data/pcaps/9.pcap'
    read_pcap_file(pcap_file_path=path, max_packet_to_read=2)