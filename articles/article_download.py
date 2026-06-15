import pandas as pd
import requests
import time
import argparse
from Bio import Entrez
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def parse_args():
    """Parses command line arguments."""
    parser = argparse.ArgumentParser(
        description="Script to collect PMIDs and abstracts for a list of proteins.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Files
    parser.add_argument('--ids-file', type=str, default='./old_train_uniprot_ids.txt',
                        help='EntryIDs list')
    parser.add_argument('--seq-file', type=str, default='./old_train_seq.feather',
                        help='Path to the Feather file containing sequences and EntryID.')
    parser.add_argument('--output-mapping', type=str, default='train_protein_to_pmid.csv',
                        help='File to save the Protein -> PMID mapping.')
    parser.add_argument('--output-abstracts', type=str, default='./old_train_final_abstracts.csv',
                        help='File to save final data with abstracts.')

    # NCBI/UniProt Settings
    parser.add_argument('--email', type=str, default='my-mail@gmai.com',
                        help='Email for NCBI registration (required for API usage).')
    
    # Batch processing parameters
    parser.add_argument('--uniprot-batch', type=int, default=50,
                        help='Batch size for UniProt requests.')
    parser.add_argument('--pubmed-batch', type=int, default=200,
                        help='Batch size for PubMed requests.')

    return parser.parse_args()

def create_session():
    """Creates a session with retry mechanism."""
    session = requests.Session()
    retries = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504, 429])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_pmids_from_uniprot(protein_ids, batch_size):
    """
    Retrieves PubMed IDs for a list of proteins by parsing the 'references' field.
    """
    base_url = "https://rest.uniprot.org/uniprotkb/search"
    session = create_session()
    
    results = []
    total = len(protein_ids)
    
    print(f"[*] Starting PMID collection for {total} proteins from UniProt (batch={batch_size})...")

    for i in range(0, total, batch_size):
        batch = protein_ids[i:i + batch_size]
        
        # Formulate query
        query = " OR ".join([f"accession:{pid}" for pid in batch])
        
        params = {
            "query": query,
            "fields": "accession,lit_pubmed_id", 
            "format": "json",
            "size": 500
        }
        
        try:
            response = session.get(base_url, params=params)
            response.raise_for_status()
            data = response.json()
            
            for entry in data.get('results', []):
                acc = entry.get('primaryAccession')
                refs_list = entry.get('references', [])
                
                found_pmids = set()

                for ref in refs_list:
                    citation = ref.get('citation', {})
                    cross_refs = citation.get('citationCrossReferences', [])
                    
                    for xref in cross_refs:
                        if xref.get('database') == 'PubMed':
                            found_pmids.add(xref.get('id'))
                
                for pmid in found_pmids:
                    results.append({'ProteinID': acc, 'PMID': pmid})

            print(f"   Processed {min(i + batch_size, total)} / {total} proteins", end='\r')
            time.sleep(0.5)
            
        except Exception as e:
            print(f"\n[!] Error in batch {i}: {e}")

    print(f"\n[*] PMID collection completed. Found {len(results)} links.")
    return pd.DataFrame(results)

def get_abstracts_from_pubmed(pmid_list, user_email, batch_size):
    """
    Downloads abstracts from PubMed (NCBI) using BioPython.
    """
    if not pmid_list:
        return pd.DataFrame()

    Entrez.email = user_email
    Entrez.tool = "PythonProteinScraper"
    
    unique_pmids = list(set(pmid_list))
    total_pmids = len(unique_pmids)
    
    print(f"[*] Starting download of {total_pmids} unique abstracts from PubMed (batch={batch_size})...")
    
    abstracts_data = []
    
    for i in range(0, total_pmids, batch_size):
        batch = unique_pmids[i:i + batch_size]
        try:
            handle = Entrez.efetch(db="pubmed", id=batch, rettype="xml", retmode="xml")
            records = Entrez.read(handle)
            handle.close()
            
            for paper in records['PubmedArticle']:
                try:
                    pmid = str(paper['MedlineCitation']['PMID'])
                    article = paper['MedlineCitation']['Article']
                    title = article.get('ArticleTitle', '')
                    
                    abstract_list = article.get('Abstract', {}).get('AbstractText', [])
                    abstract_text = " ".join([str(x) for x in abstract_list])
                    
                    abstracts_data.append({
                        'PMID': pmid,
                        'Title': title,
                        'Abstract': abstract_text
                    })
                except KeyError:
                    continue 
            
            print(f"   Downloaded {min(i + batch_size, total_pmids)} / {total_pmids} articles", end='\r')
            time.sleep(1) 
            
        except Exception as e:
           print(f"\n[!] Error downloading from PubMed (batch {i}): {e}")
           time.sleep(5)

    print(f"\n[*] Abstract download completed.")
    return pd.DataFrame(abstracts_data)

def main():
    args = parse_args()

    # Read input data
    try:
        train = pd.read_feather(args.seq_file)
        proteins = train['EntryID'].values.tolist()
    except FileNotFoundError:
        print(f"[!] Error: File '{args.seq_file}' not found.")
        return
    except KeyError:
        print(f"[!] Error: Column 'EntryID' missing in file '{args.seq_file}'.")
        return

    with open(args.ids_file, 'wt') as f:
        f.write('\n'.join(proteins))

    # 2. Get PMIDs from UniProt
    df_links = get_pmids_from_uniprot(proteins, args.uniprot_batch)
    
    if df_links.empty:
        print("No PMIDs found. Check the protein list or connection.")
        return

    df_links.to_csv(args.output_mapping, index=False)
    print(f"[+] Mapping saved to {args.output_mapping}")
    
    # 3. Download texts from PubMed
    all_pmids = df_links['PMID'].tolist()
    df_abstracts = get_abstracts_from_pubmed(all_pmids, args.email, args.pubmed_batch)
    
    if df_abstracts.empty:
        print("Failed to download abstracts.")
        return

    # 4. Merge data
    final_df = pd.merge(df_links, df_abstracts, on='PMID', how='left')
    
    final_df.to_csv(args.output_abstracts, index=False)
    print(f"\n[SUCCESS] Done! Results saved to {args.output_abstracts}")

if __name__ == "__main__":
    main()

