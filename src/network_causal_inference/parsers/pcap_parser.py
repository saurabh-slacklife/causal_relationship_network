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
    for i, packet in PcapReader(pcap_file_path):
        try:
            print(packet.summary())
        except KeyboardInterrupt:
            print('Interrupted by user')
            raise
        except Exception:
            raise


if __name__ == '__main__':
    path = '../../../data/pcaps/9.pcap'
    read_pcap_file(pcap_file_path=path, max_packet_to_read=2)



# def parse_pcap_data(pcap_file_path: str, max_packets: int =10) -> DataFrame:
#     packets_data = list
#
#     try:
#         packets = rdpcap(pcap_file_path)
#
#         if max_packets:
#             packets = packets[:max_packets]
#
#         for _, pckt in enumerate(packets):
#             packet_info = {
#                 'timestamp': pckt.time,
#                 'length': len(pckt),
#                 'protocol': pckt.name,
#                 'src_ip': None,
#                 'dst_ip': None,
#                 'src_port': None,
#                 'dst_port': None,
#                 'ttl': None
#             }
#
#             # Extract IP layer information
#             if pckt.haslayer(IP):
#                 ip_layer = pckt[IP]
#                 packet_info['src_ip'] = ip_layer.src
#                 packet_info['dst_ip'] = ip_layer.dst
#                 packet_info['ttl'] = ip_layer.ttl
#
#                 # Extract transport layer information
#             elif pckt.haslayer(TCP):
#                 tcp_layer = pckt[TCP]
#                 packet_info['src_port'] = tcp_layer.sport
#                 packet_info['dst_port'] = tcp_layer.dport
#
#             elif pckt.haslayer(UDP):
#                 udp_layer = pckt[UDP]
#                 packet_info['src_port'] = udp_layer.sport                    packet_info['dst_port'] = udp_layer.dport
#
#             packets_data.append(packet_info)
#
#             if (i + 1) % 1000 == 0:
#                 print(f"Processed {i + 1} packets...")
#
#         df = pd.DataFrame(packets_data)
#         print(f"Successfully parsed {len(df)} packets")
#         return df
#
#     except Exception as e:
#         print(f"Error parsing pcap: {e}")
#         return pd.DataFrame()


# Batch processing for multiple UNSW-NB15 pcap files
def batch_process_unswnb15(pcap_dir, output_dir="parsed_data"):
    """
    Process all PCAP files in a directory
    Args:
        pcap_dir: Directory containing UNSW-NB15 pcap files
        output_dir: Directory to save parsed CSV files
    """
    import os
    from pathlib import Path

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all pcap files
    pcap_files = list(Path(pcap_dir).glob("*.pcap")) + \
                 list(Path(pcap_dir).glob("*.pcapng"))

    print(f"Found {len(pcap_files)} pcap files")

    for pcap_file in pcap_files:
        print(f"\nProcessing {pcap_file.name}...")

        # Parse the pcap file
        df = parse_pcap_scapy(str(pcap_file), max_packets=None)

        if not df.empty:
            # Save to CSV
            output_file = Path(output_dir) / f"{pcap_file.stem}_parsed.csv"
            df.to_csv(output_file, index=False)
            print(f"Saved to {output_file}")

            # Basic analysis
            print(f"  Total packets: {len(df)}")
            print(f"  Unique source IPs: {df['src_ip'].nunique()}")
            print(f"  Unique destination IPs: {df['dst_ip'].nunique()}")
            print(f"  Protocols: {df['protocol'].value_counts().to_dict()}")


# Example usage for batch processing
if __name__ == "__main__":
    # Process all pcap files in a directory
    pcap_directory = "path/to/UNSW-NB15/pcap/files"
    batch_process_unswnb15(pcap_directory)