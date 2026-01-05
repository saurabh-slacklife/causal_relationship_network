from scapy.all import *
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
    for pckt in PcapReader(pcap_file_path):
        try:
            logger.info('********** packet ***********: \n %s',pckt.summary())
        except KeyboardInterrupt:
            print('Interrupted')
            raise
        except Exception:
            raise

def parse_pcap_data(pcap_file_path: str, max_packets: int =10) -> DataFrame:
    packets_data = list

    try:
        packets = rdpcap(pcap_file_path)

        if max_packets:
            packets = packets[:max_packets]

        for _, pckt in enumerate(packets):
            packet_info = {
                'timestamp': pckt.time,
                'length': len(pckt),
                'protocol': pckt.name,
                'src_ip': None,
                'dst_ip': None,
                'src_port': None,
                'dst_port': None,
                'ttl': None
            }

            if pckt.haslayer('IP'):
                ip_layer = pckt['IP']
                packet_info['src_ip'] = ip_layer.src
                packet_info['dst_ip'] = ip_layer.dst
                packet_info['ttl'] = ip_layer.ttl


            packets_data.append(packet_info)

        df = pd.DataFrame(packets_data)
        logger.info('********** packet dtaraframe ***********: \n %s', df)
        return df

    except Exception as e:
        logger.error('error trace:%s',e)
        return pd.DataFrame()

if __name__ == '__main__':
    path = '../../../data/pcaps/9.pcap'
    read_pcap_file(pcap_file_path=path, max_packet_to_read=2)