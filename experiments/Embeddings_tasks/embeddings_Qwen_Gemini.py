"""Embedding pipeline driven by a *dataset‑spec* list.

Each element of ``dataset_specs`` is a **5‑tuple**::

    (path, name, task, type, extra_dict, extra_list)

* **path** – str or ``Path`` to the pickle file
* **name** – logical dataset identifier (e.g. "FOLIO")
* **task** – what you will later group by (e.g. "translation_ranking")
* **type** – modality label such as "NL" or "FOL"
* **extra_dict** – add extra dict nesting structure for compatibility
* **extra_list** – add extra list nesting structure for compatibility

Run in two modes (``compute`` and ``evaluate``) selected via a **positional**
command‑line argument:

```
python embedding_pipeline.py compute   # original behaviour
python embedding_pipeline.py evaluate  # load artefacts and run evaluation
```
"""
from __future__ import annotations

################################################################################
# Imports & typing helpers
################################################################################

import argparse
import copy
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from typing import List
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput

################################################################################
# Data containers
################################################################################


@dataclass
class Dataset:
    """Wrapper around a stories dict plus minimal metadata."""

    name: str
    task: str
    type: str
    stories: Mapping  # nested list‑of‑lists structure loaded from pickle
    inverted: Mapping

    def __init__(
        self,
        name: str,
        task: str,
        type: str,
        stories: Mapping,
        extra_dict: bool,
        extra_list: bool,
    ) -> None:
        self.name = name
        self.task = task
        self.type = type

        # Harmonise nested layout ------------------------------------------------
        if extra_dict:
            self.stories = {0: stories}
        else:
            self.stories = stories

        if extra_list:
            for k in self.stories:
                self.stories[k] = [[s] for s in self.stories[k]]

        # Build inverted index for constant‑time look‑ups ------------------------
        self.inverted = self.build_inverted_index(self.stories)

    # --------------------------------------------------------------------- utils
    @staticmethod
    def build_flat_index(stories: Mapping) -> Dict[int, List[int]]:  # noqa: D401
        """Return story‑local indices flattened to a single running counter."""
        index: Dict[int, List[int]] = {}
        next_id = 0
        for story_id, sentences in stories.items():
            index[story_id] = []
            for mods_in_sentence in sentences:
                sent_idx: List[int] = []
                for _ in mods_in_sentence:
                    sent_idx.append(next_id)
                    next_id += 1
                index[story_id].append(sent_idx)
        return index

    @staticmethod
    def build_inverted_index(stories: Mapping) -> Dict[int, tuple]:  # noqa: D401
        """Map each flat index back to (story_id, sentence_idx, mod_idx)."""
        inverted: Dict[int, tuple] = {}
        next_id = 0
        for story_id, sentences in stories.items():
            for sent_idx, mods_in_sentence in enumerate(sentences):
                for mod_idx, _ in enumerate(mods_in_sentence):
                    inverted[next_id] = (story_id, sent_idx, mod_idx)
                    next_id += 1
        return inverted


@dataclass
class Prompt:
    name: str
    text: str | None


################################################################################
# Embedding helper
################################################################################

def gemini_embed(
    texts: List[str],
    *,
    project: str | None = None,
    location: str = "us-central1",
    model_name: str = "gemini-embedding-001",
    task_type: str | None = 'SEMANTIC_SIMILARITY',           # e.g. "RETRIEVAL_DOCUMENT"
    batch_size: int = 512,
) -> torch.Tensor:
    """
    Batch–embed *texts* with a Gemini model hosted on Vertex AI and
    return a 2‑D L2‑normalised torch.Tensor.
    """

    print("ATTENTON - GEMINI IS RUNNING!")

    # ① Initialise Vertex AI once per process
    aiplatform.init(project=project, location=location)

    model = TextEmbeddingModel.from_pretrained(model_name)  # 3072‑D for gemini‑embedding‑001 :contentReference[oaicite:3]{index=3}

    embeddings: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]

        # Build TextEmbeddingInput objects (task_type optional) :contentReference[oaicite:4]{index=4}
        inputs = [
            TextEmbeddingInput(text=t, task_type=task_type)  # type: ignore[arg-type]
            for t in chunk
        ]

        resp = model.get_embeddings(inputs)                 # → List[Embedding]
        embeddings.extend([r.values for r in resp])

    # Convert to Torch and normalise exactly like the Qwen path
    tens = torch.tensor(embeddings, dtype=torch.float32)
    return torch.nn.functional.normalize(tens, p=2, dim=1)
    

def embed(
    stories: Mapping,
    inverted: Mapping,
    *,
    model: SentenceTransformer | None = None,
    provider: str = "qwen",     # "qwen" (default) or "gemini"
    project: str | None = None, # only used by Gemini
    location: str = "us-central1",
    batch_size: int = 64,
    prompt: str | None = None,
    max_examples: int | None = None,
) -> torch.Tensor:
    """
    Encode all modifications in *stories* with either a local
    SentenceTransformer ('qwen') or a remote Gemini embedding model.
    """

    # Re‑use existing code to linearise the mods
    mods: List[str] = [
        stories[s_id][sent_i][mod_i]
        for idx in range(len(inverted))
        for (s_id, sent_i, mod_i) in (inverted[idx],)
    ]
    if max_examples is not None:
        mods = mods[:max_examples]

    if provider == "qwen":
        if model is None:
            raise ValueError("Local provider requires a SentenceTransformer instance.")
        with torch.no_grad():
            emb = (
                model.encode(
                    mods,
                    prompt=prompt,
                    batch_size=batch_size,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                )
                .cpu()
            )

    elif provider == "gemini":
        # Delegate to the helper; Gemini models ignore *prompt*
        emb = gemini_embed(
            mods,
            project=project,
            location=location,
            batch_size=batch_size,
        )

    else:
        raise ValueError(f"Unknown provider '{provider}'. Choose 'qwen' or 'gemini'.")

    return emb


################################################################################
# Evaluation helper
################################################################################


def evaluate_ranking_and_sim(
    task: str,
    stories: Mapping,
    inverted_stories_results: torch.Tensor,
    inverted: Mapping,
    stories_gt: Mapping,
    inverted_stories_results_gt: torch.Tensor,
    inverted_gt: Mapping,
) -> Dict[str, List[bool]]:
    """Return ranking correctness for *equiv*, *neg* and *both* tasks."""

    # Re‑insert embeddings into the original nested structures ---------------
    stories_results = copy.deepcopy(stories)
    for idx, value in inverted.items():
        stories_results[value[0]][value[1]][value[2]] = inverted_stories_results[idx]

    stories_results_gt = copy.deepcopy(stories_gt)
    for idx, value in inverted_gt.items():
        stories_results_gt[value[0]][value[1]][value[2]] = (
            inverted_stories_results_gt[idx]
        )

    #breakpoint()

    if "ranking" in task:
        results = {"equiv": [], "neg": [], "both": []}
    else:
        results = {"sim": []}
        
    for story_id, sentences in stories_results.items():
        for sent_idx, sentence in enumerate(sentences):
            sims = F.cosine_similarity(
                stories_results_gt[story_id][sent_idx][0]
                .unsqueeze(0)
                .expand(len(sentence), -1),
                torch.stack(sentence),
                dim=1,
            )
            ranking = torch.argsort(sims, descending=True)

            if "ranking" in task:
                good_equiv = (ranking[0].item() in {0, 1}) and (ranking[1].item() in {0, 1})
                good_neg = (
                    ranking[-1].item() in {len(ranking) - 1, len(ranking) - 2}
                    and ranking[-2].item() in {len(ranking) - 1, len(ranking) - 2}
                )
                results["equiv"].append(good_equiv)
                results["neg"].append(good_neg)
                results["both"].append(good_equiv and good_neg)
            else:
                good_sim = (ranking[0].item() in {0})
                results["sim"].append(good_sim)                  

    return results


################################################################################
# COMPUTE MODE (original behaviour)
################################################################################

def compute(
    *,
    save_dir: Path = Path("./Embedding"),
    args = None,
) -> None:
    """Reproduce the original embedding‑extraction pipeline."""

    # ---------------------------------------------------------------- PROMPTS
    prompts: List[Prompt] = [
        Prompt(name="None", text=None),
#        Prompt(
#            name="logic‑NL",
#            text=(
#                "Instruct: Encode the first‑order logic meaning of the "
#                "following natural‑language sentence.\nSentence:"
#            ),
#        ),
#        Prompt(
#            name="logic‑FOL",
#            text=(
#                "Instruct: Encode the first‑order logic meaning of the "
#                "following first‑order formula.\nSentence:"
#            ),
#        ),
    ]

    # ---------------------------------------------------------- DATASET SPECS
    dataset_specs: Sequence[Tuple[str | Path, str, str, str, bool, bool]] = [
       (
           save_dir / "Translation_ranking_complete_FOLIO.pkl",
           "FOLIO",
           "translation_ranking",
           "NL",
           False,
           False,
       ),
       (
           save_dir / "list_modifications_ranking_FOLIO.pkl",
           "FOLIO",
           "translation_ranking",
           "FOL",
           False,
           False,
       ),
       (
           save_dir / "ground-truths_FOLIO.pkl",
           "FOLIO",
           "GT",
           "NL",
           False,
           True,
       ),
    # UNCOMMENT THE FOLLOWING TWO ELEMENTS AND COMMENT THE FIRST TWO TO SWITCH BETWEEN RANKING AND MOST SIMILAR TASKS
    #    (
    #        save_dir / "Translation_most_similar_FOLIO.pkl",
    #        "FOLIO",
    #        "most_similar",
    #        "NL",
    #        False,
    #        False,
    #    ),
    #    (
    #        save_dir / "list_modifications_most_similar_FOLIO.pkl",
    #        "FOLIO",
    #        "most_similar",
    #        "FOL",
    #        False,
    #        False,
    #    )
    ]

    # ------------------------------------------------------------ LOADING
    datasets: List[Dataset] = []
    for pth, name, task, typ, extra_dict, extra_list in dataset_specs:
        with Path(pth).expanduser().open("rb") as fh:
            stories = pickle.load(fh)
        datasets.append(
            Dataset(
                name=name,
                task=task,
                type=typ,
                stories=stories,
                extra_dict=extra_dict,
                extra_list=extra_list,
            )
        )

    # ------------------------------------------------------------ MODEL SETUP
    if args.provider == "qwen":
        model = SentenceTransformer("Qwen/Qwen3-Embedding-8B", device="cuda")
        model.eval()

    # ------------------------------------------------------------- PROCESSING
    results: Dict[str, Dict[str, Dict[str, Dict[str, torch.Tensor]]]] = {}

    for ds in datasets:
        for prm in prompts:
            if ds.type in prm.name or prm.name == "None":
                emb = embed(
                    ds.stories,
                    ds.inverted,
                    model=model if args.provider == "qwen" else None,
                    prompt=prm.text,
                    provider=args.provider,
                    project=args.gcp_project,
                    location=args.gcp_location,
                    batch_size=128,
                    max_examples=args.max_examples,
                )
                (
                    results.setdefault(ds.name, {})
                    .setdefault(ds.task, {})
                    .setdefault(ds.type, {})[prm.name]
                ) = emb
                print(
                    f"✓ {ds.name}/{ds.task}/{ds.type} with prompt '{prm.name}': "
                    f"shape {tuple(emb.shape)}"
                )
            else:
                print(
                    f"Skipping {ds.name}/{ds.task}/{ds.type} with prompt '{prm.name}'"
                )

    # -------------------------------------------------------------- SAVE FILES
    embeddings_file = save_dir / "all_embeddings.pt"
    torch.save(results, embeddings_file)
    print(f"All embeddings saved to '{embeddings_file}'.")

    datasets_file = save_dir / "datasets_formatted.pkl"
    with datasets_file.open("wb") as f:
        pickle.dump(datasets, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Datasets saved to '{datasets_file}'.")


################################################################################
# EVALUATE MODE
################################################################################

def evaluate(
    *,
    embeddings_file: Path = Path("./Embedding/all_embeddings.pt"),
    datasets_file: Path = Path("./Embedding/datasets_formatted.pkl"),
) -> None:
    """Load persisted artefacts and run `evaluate_ranking`."""

    if not embeddings_file.exists() or not datasets_file.exists():
        raise FileNotFoundError(
            "Required artefacts not found. Run in 'compute' mode first."
        )

    # ------------------------------------------------------- LOAD ARTEFACTS
    all_embeddings: Dict = torch.load(embeddings_file, map_location="cpu")
    with datasets_file.open("rb") as fh:
        datasets: List[Dataset] = pickle.load(fh)

    # ------------------------------------------------------------ EVALUATION
    ds_lookup = {(d.name, d.task, d.type): d for d in datasets}

    overall_scores = []
    print("Evaluating datasets…")

    for ds in datasets:
        if "GT" in ds.task:
            continue
            
        # Locate matching ground‑truth dataset (type/format independent)
        gt_ds = ds_lookup.get((ds.name, "GT", "NL"))
        if gt_ds is None:
            print(f"⚠ No GT dataset found for '{ds.name}'. Skipping.")
            continue

        try:
            for prompt_type in all_embeddings[ds.name][ds.task][ds.type]:
                src_emb = all_embeddings[ds.name][ds.task][ds.type][prompt_type]
                for gt_prompt_type in ["None", "logic‑NL"]:
                    gt_emb = all_embeddings[gt_ds.name][gt_ds.task][gt_ds.type][gt_prompt_type]
        
                    res = evaluate_ranking_and_sim(
                        ds.task,
                        ds.stories,
                        src_emb,
                        ds.inverted,
                        gt_ds.stories,
                        gt_emb,
                        gt_ds.inverted,
                    )

                    if "ranking" in ds.task:
                        # Compute simple accuracies ------------------------------------------------
                        equiv_acc = sum(res["equiv"]) / len(res["equiv"])
                        neg_acc = sum(res["neg"]) / len(res["neg"])
                        both_acc = sum(res["both"]) / len(res["both"])
                
                        overall_scores.append((equiv_acc, neg_acc, both_acc))
                        print(
                            f"✓ {ds.name}/{ds.task}/{ds.type}/{prompt_type} with {gt_ds.name}/{gt_ds.task}/{gt_ds.type}/{gt_prompt_type} equiv={equiv_acc:.2%} | "
                            f"neg={neg_acc:.2%} | both={both_acc:.2%}"
                        )
                    else:
                        # Compute simple accuracies ------------------------------------------------
                        sim_acc = sum(res["sim"]) / len(res["sim"])
                
                        overall_scores.append((sim_acc))
                        print(
                            f"✓ {ds.name}/{ds.task}/{ds.type}/{prompt_type} with {gt_ds.name}/{gt_ds.task}/{gt_ds.type}/{gt_prompt_type} most_sim={sim_acc:.2%} "
                        )
        except KeyError:
            print(
                f"⚠ Embeddings missing for '{ds.name}/{ds.task}/{ds.type}'. Skipping."
            )
            continue



    if overall_scores:
        if (isinstance(overall_scores, list) and  isinstance(overall_scores[0], tuple)) or isinstance(overall_scores, tuple):
            mean_equiv = sum(s[0] for s in overall_scores) / len(overall_scores)
            mean_neg = sum(s[1] for s in overall_scores) / len(overall_scores)
            mean_both = sum(s[2] for s in overall_scores) / len(overall_scores)
            print(
                "------------------------------\n" +
                f"  AVERAGE  equiv={mean_equiv:.2%} | "
                f"neg={mean_neg:.2%} | both={mean_both:.2%}\n" +
                "------------------------------"
            )
        else:
            mean_sim = sum(s for s in overall_scores) / len(overall_scores)
            print(
                "------------------------------\n" +
                f"  AVERAGE  most_sim={mean_sim:.2%}\n " +
                "------------------------------"
            )
    else:
        print("No datasets were evaluated.")


################################################################################
# Entrypoint
################################################################################

def main() -> None:
    """Parse CLI and dispatch to the selected mode."""

    parser = argparse.ArgumentParser(
        description="Embedding & evaluation pipeline with two modes.",
    )
    parser.add_argument(
        "mode",
        choices=["compute", "evaluate"],
        help="Operation mode. 'compute' reproduces the original pipeline; "
        "'evaluate' loads saved artefacts and runs evaluation.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of examples per dataset (compute mode).",
    )
    parser.add_argument(
        "--provider",
        choices=["qwen", "gemini"],
        default="qwen",
        help="Embedding backend to use.",
    )
    parser.add_argument(
        "--gcp-project",
        type=str,
        default=None,
        help="(Gemini only) Your Google Cloud project ID."
    )
    parser.add_argument(
        "--gcp-location",
        type=str,
        default="us-central1",
        help="(Gemini only) Region of the Vertex AI endpoint."
    )
    args = parser.parse_args()

    if args.mode == "compute":
        compute(args=args)
    elif args.mode == "evaluate":
        evaluate()
    else:  # This should never happen due to argparse's choices
        parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
