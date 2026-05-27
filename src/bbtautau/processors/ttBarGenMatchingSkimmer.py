"""
Skimmer for TTbar background study with top-b GenJet matching.
Based on https://github.com/LPC-HH/HH4b/blob/main/src/HH4b/processors/bbbbSkimmer.py.

Author(s): Claudio Quaranta
"""

from __future__ import annotations

import logging
import pathlib
import time
from collections import OrderedDict
from typing import ClassVar

import awkward as ak
import numpy as np
from boostedhh import hh_vars
from boostedhh.processors import SkimmerABC, utils
from boostedhh.processors.corrections import (
    JECs,
    add_pileup_weight,
    add_ps_weight,
    get_jetveto_event,
    get_pdf_weights,
    get_scale_weights,
)
from boostedhh.processors.utils import (
    P4,
    PAD_VAL,
    add_selection,
    pad_val,
)
from coffea import processor
from coffea.analysis_tools import PackedSelection, Weights

from bbtautau.HLTs import HLTs

from . import GenSelection, objects

# mapping samples to the appropriate function for doing gen-level selections
gen_selection_dict = {
    "HHto4B": GenSelection.gen_selection_HH4b,
    "HHto2B2Tau": GenSelection.gen_selection_HHbbtautau,
}

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

package_path = str(pathlib.Path(__file__).parent.parent.resolve())


class ttBarGenMatchingSkimmer(SkimmerABC):
    """
    Skims nanoaod files, saving selected branches and events passing preselection cuts
    (and triggers for data).
    """

    # name in nano files: name in the skimmed output
    skim_vars = {  # noqa: RUF012
        "Jet": {
            **P4,
            "rawFactor": "rawFactor",
            "btagPNetB": "btagPNetB",
            "btagDeepFlavB": "btagDeepFlavB",
        },
        "MET": {
            "pt": "Pt",
            "phi": "Phi",
        },
        "Lepton": {
            **P4,
            "charge": "charge",
        },
        "Tau": {
            **P4,
            "charge": "charge",
            "idDeepTau2018v2p5VSjet": "DeepTauvsJet",
            "idDeepTau2018v2p5VSmu": "DeepTauvsMu",
            "idDeepTau2018v2p5VSe": "DeepTauvsE",
        },
        "BoostedTau": {
            **P4,
            "charge": "charge",
            "idMVAnewDM2017v2": "idMVAnewDM2017v2",
        },
        "SubJet": {
            **P4,
        },
        "FatJet": {
            **P4,
            "msoftdrop": "Msd",
            "t32": "Tau3OverTau2",
            "rawFactor": "rawFactor",
            "particleNet_massCorr": "particleNet_massCorr",
            # tagger variables added below
        },
        "GenHiggs": P4,
        "Event": {
            "run": "run",
            "event": "event",
            "luminosityBlock": "luminosityBlock",
        },
        "Pileup": {
            "nPU",
        },
        "TriggerObject": {
            "pt": "Pt",
            "eta": "Eta",
            "phi": "Phi",
            "filterBits": "Bit",
        },
    }

    # only applied if fatjet_bb_preselection is True
    preselection = {  # noqa: RUF012
        # roughly, 85% signal efficiency, 2% QCD efficiency (pT: 250-400, mSD:0-250, mRegLegacy:40-250)
        "pnet-legacy": 0.8,
        "pnet-v12": 0.3,
        "glopart-v2": 0.3,
    }

    fatjet_selection = {  # noqa: RUF012
        "_object_pt": 170,
        "pt": 200,
        "eta": 2.5,
        "mass": 50,
        "msd": 0,
        "mreg": 0,
    }

    lepton_selection: ClassVar[dict[str, float]] = {
        "ptcut": 30,
        "etacut": 2.5,
        "dzcut": 0.2,
        "dxycut": 0.045,
    }

    vbf_jet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 4.7,
        "id": "tight",
        "dr_fatjets": 1.2,
        "dr_leptons": 0.4,
    }

    vbf_veto_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
    }

    ak4_bjet_selection = {  # noqa: RUF012
        "pt": 25,
        "eta_max": 2.5,
        "id": "tight",
        "dr_fatjets": 1.0,
        "dr_leptons": 0.4,
    }

    ak4_bjet_lepton_selection = {  # noqa: RUF012
        "electron_pt": 5,
        "muon_pt": 7,
    }

    def __init__(
        self,
        xsecs: dict = None,
        save_systematics: bool = False,
        region: str = "signal",
        nano_version: str = "v12_private",
        fatjet_pt_cut: float = None,
        fatjet_bb_preselection: bool = False,
        prescale_factor: int = None,
        genjet_topb_dr: float = 0.4,
        ak4_genjet_topb_dr: float = 0.4,
        num_gen_top_b: int = 2,
        num_genjet_top_b: int = 4,
    ):
        super().__init__()

        self.XSECS = xsecs if xsecs is not None else {}  # in pb

        # HLT selection
        self.HLTs = {"signal": HLTs.hlt_list(hlt_prefix=False)}
        self.HLTs = self.HLTs[region]
        self._systematics = save_systematics
        self._nano_version = nano_version
        self._region = region
        self._accumulator = processor.dict_accumulator({})
        self._fatjet_bb_preselection = fatjet_bb_preselection
        self._prescale_factor = prescale_factor
        self._genjet_topb_dr = genjet_topb_dr
        self._ak4_genjet_topb_dr = ak4_genjet_topb_dr
        self._num_gen_top_b = num_gen_top_b
        self._num_genjet_top_b = num_genjet_top_b
        self._chunk_counter = 0

        # JMSR
        self.jmsr_vars = ["msoftdrop", "particleNet_mass_legacy", "ParTmassVis", "ParTmassRes"]

        # particlenet legacy variables
        pnet_vars = [
            "Xbb",
            "QCD",
            "QCDb",
            "QCDbb",
            "QCDcc",
            "QCDc",
            "QCDothers",
            "XbbvsQCD",
            "mass",
        ]
        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"particleNetLegacy_{var}": f"PNet{var}Legacy" for var in pnet_vars},
        }

        # glopart variables
        glopart_vars = [
            "QCD1HF",
            "QCD2HF",
            "QCD0HF",
            "TopW",
            "TopbW",
            "TopbWev",
            "TopbWmv",
            "TopbWtauhv",
            "TopbWq",
            "TopbWqq",
            "Xbb",
            "Xcc",
            "Xcs",
            "Xgg",
            "Xqq",
            "Xtauhtaue",
            "Xtauhtauh",
            "Xtauhtaum",
            # Derived variables
            "massResCorr",
            "massVisCorr",
            "massResApplied",
            "massVisApplied",
            "QCD",
            "Top",
            "XbbvsQCD",
            "XbbvsQCDTop",
            "XtauhtauevsQCD",
            "XtauhtauevsQCDTop",
            "XtauhtaumvsQCD",
            "XtauhtaumvsQCDTop",
            "XtauhtauhvsQCD",
            "XtauhtauhvsQCDTop",
        ]

        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"globalParT_{var}": f"ParT{var}" for var in glopart_vars},
        }

        # CA variables
        ca_vars = [
            "tau_number",
            "tau_number_in_fatjet",
            "globalParT_massVisApplied_oneHPSTau",
            "globalParT_massVisApplied_oneHPSTau_thth",
            "globalParT_massVisApplied_oneHPSTauorMuon_thtm",
            "globalParT_massVisApplied_oneHPSTauorElectron_thte",
            "globalParT_massVisApplied_with_delta_axis_merged",
            "globalParT_massVisApplied_oneHPSTauorLepton_flag",
            "globalParT_massVisApplied_000_fatjetwithMET",
            "globalParT_massVisApplied_000_fatjet",
            "globalParT_massVisApplied_000_fatjet_MET_with_same_dirc",
            "mass_merged",
            "msoftdrop_merged",
            "globalParT_massVisApplied_merged",
            "globalParT_massResApplied_merged",
            "particleNet_mass_legacy_merged",
            "Tauflag",
            "one_elec_in_fatjet",
            "one_muon_in_fatjet",
            "one_elec",
            "one_muon",
            "mass_fatjet_et",
            "mass_fatjet_mt",
            "isDauTau",
            "mass",
            "msoftdrop",
            "globalParT_massVisApplied",
            "globalParT_massResApplied",
            "particleNet_mass_legacy",
            "dau0_pt",
            "dau1_pt",
            "dau0_eta",
            "dau1_eta",
            "dau0_phi",
            "dau1_phi",
            "dau0_mass",
            "dau1_mass",
            "ntaus_perfatjets",
            "mass_subjets",
            "mass_boostedtaus",
            "nsubjets_perfatjets",
            "mass_fatjets",
            "mass_mt",
            "msoftdrop_mt",
            "globalParT_massVisApplied_mt",
            "globalParT_massResApplied_mt",
            "particleNet_mass_legacy_mt",
            "isDauTau_mt",
            "dau0_pt_mt",
            "dau1_pt_mt",
            "dau0_eta_mt",
            "dau1_eta_mt",
            "dau0_phi_mt",
            "dau1_phi_mt",
            "dau0_mass_mt",
            "dau1_mass_mt",
            "ntaus_perfatjets_mt",
            "mass_subjets_mt",
            "mass_boostedtaus_mt",
            "nsubjets_perfatjets_mt",
            "mass_subjets_mt_01",
            "muon_subjet_dr02",
            "mass_subjets_mt_1",
            "mass_subjets_mt_0",
            "mass_et",
            "msoftdrop_et",
            "globalParT_massVisApplied_et",
            "globalParT_massResApplied_et",
            "particleNet_mass_legacy_et",
            "isDauTau_et",
            "dau0_pt_et",
            "dau1_pt_et",
            "dau0_eta_et",
            "dau1_eta_et",
            "dau0_phi_et",
            "dau1_phi_et",
            "dau0_mass_et",
            "dau1_mass_et",
            "ntaus_perfatjets_et",
            "mass_subjets_et",
            "mass_boostedtaus_et",
            "nsubjets_perfatjets_et",
            "mass_subjets_et_01",
            "elec_subjet_dr02",
            "mass_subjets_et_1",
            "mass_subjets_et_0",
        ]

        self.skim_vars["FatJet"] = {
            **self.skim_vars["FatJet"],
            **{f"CA_{var}": f"CA{var}" for var in ca_vars},
        }

        # update fatjet pT cut
        if fatjet_pt_cut is not None:
            self.fatjet_selection["pt"] = fatjet_pt_cut

        logger.info(
            f"Running skimmer with:\nsystematics {self._systematics}\nregion {self._region}\nfatjet pt cut {self.fatjet_selection['pt']}"
        )

    @property
    def accumulator(self):
        return self._accumulator

    def _empty_ttbar_gen_matching_vars(self, n_events: int, num_ak4_jets: int) -> dict:
        """Return PAD_VAL-filled gen/matching branches for data or samples without gen info."""
        return {
            "nGenTopB": np.zeros(n_events, dtype=np.int32),
            "nGenJetFromTopB": np.zeros(n_events, dtype=np.int32),
            "GenTopBPt": np.full((n_events, self._num_gen_top_b), PAD_VAL),
            "GenTopBEta": np.full((n_events, self._num_gen_top_b), PAD_VAL),
            "GenTopBPhi": np.full((n_events, self._num_gen_top_b), PAD_VAL),
            "GenTopBMass": np.full((n_events, self._num_gen_top_b), PAD_VAL),
            "GenTopBParentPdgId": np.full((n_events, self._num_gen_top_b), PAD_VAL),
            "GenJetFromTopBPt": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBEta": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBPhi": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBMass": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBIdx": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBPartonFlavour": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBHadronFlavour": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBMatchedGenBdr": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "GenJetFromTopBParentPdgId": np.full((n_events, self._num_genjet_top_b), PAD_VAL),
            "ak4JetIsMatchedToTopBGenJet": np.zeros((n_events, num_ak4_jets), dtype=np.int32),
            "ak4JetMatchedTopBGenJetPt": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetEta": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetPhi": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetMass": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetIdx": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetDR": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBParentPdgId": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetPartonFlavour": np.full((n_events, num_ak4_jets), PAD_VAL),
            "ak4JetMatchedTopBGenJetHadronFlavour": np.full((n_events, num_ak4_jets), PAD_VAL),
        }

    def get_ttbar_gen_matching_vars(
        self, events: ak.Array, jets: ak.Array, num_ak4_jets: int
    ) -> dict:
        """Build gen-level ttbar b branches and AK4 matching branches.

        Definition used here:
        1. Gen top-b quarks are GenPart with |pdgId| == 5, fromHardProcess, isLastCopy,
           and |distinctParent.pdgId| == 6.
        2. A GenJet is tagged as from top-b hadronization if its nearest such b quark is
           within self._genjet_topb_dr. If GenJet.partonFlavour exists, we additionally
           require |partonFlavour| == 5.
        3. Reco AK4 jets are matched only to these selected GenJets using nearest ΔR within
           self._ak4_genjet_topb_dr.
        """
        n_events = len(events)
        if "GenPart" not in events.fields or "GenJet" not in events.fields:
            return self._empty_ttbar_gen_matching_vars(n_events, num_ak4_jets)

        genparts = events.GenPart
        genjets = events.GenJet

        # b quarks from t -> bW.
        # IMPORTANT: distinctParent can be None for GenPart entries without a valid parent.
        # If those None values are left inside the boolean mask, awkward keeps missing
        # elements in the sliced collection; ak.num then counts them, while the saved pt
        # becomes PAD_VAL.  Fill missing parents/mask entries explicitly.
        parent_pdgid = ak.fill_none(genparts.distinctParent.pdgId, 0)
        is_top_b = (
            (abs(genparts.pdgId) == 5)
            & genparts.hasFlags(["fromHardProcess", "isLastCopy"])
            & (abs(parent_pdgid) == 6)
        )
        is_top_b = ak.fill_none(is_top_b, False)

        gen_top_b = genparts[is_top_b]
        gen_top_b = ak.with_field(
            gen_top_b,
            ak.fill_none(gen_top_b.distinctParent.pdgId, 0),
            "topParentPdgId",
        )

        # Stable ordering for the saved GenTopB0/1 columns.
        gen_top_b = gen_top_b[ak.argsort(gen_top_b.pt, axis=1, ascending=False)]

        # Keep the original GenJet local index before slicing, useful after flattening/padding.
        genjets = ak.with_field(genjets, ak.local_index(genjets, axis=1), "genJetIdx")

        matched_b_to_genjet, genjet_genb_dr = genjets.nearest(
            gen_top_b,
            axis=1,
            return_metric=True,
            threshold=self._genjet_topb_dr,
        )
        is_genjet_from_top_b = ak.fill_none(genjet_genb_dr < self._genjet_topb_dr, False)

        if "partonFlavour" in genjets.fields:
            is_genjet_from_top_b = is_genjet_from_top_b & (abs(genjets.partonFlavour) == 5)

        genjets_from_top_b = genjets[is_genjet_from_top_b]
        matched_b_for_topb_genjets = matched_b_to_genjet[is_genjet_from_top_b]
        genjet_genb_dr = genjet_genb_dr[is_genjet_from_top_b]

        genjets_from_top_b = ak.with_field(
            genjets_from_top_b,
            matched_b_for_topb_genjets.topParentPdgId,
            "topParentPdgId",
        )
        genjets_from_top_b = ak.with_field(
            genjets_from_top_b,
            genjet_genb_dr,
            "matchedGenBdr",
        )

        matched_topb_genjet, ak4_topb_genjet_dr = jets.nearest(
            genjets_from_top_b,
            axis=1,
            return_metric=True,
            threshold=self._ak4_genjet_topb_dr,
        )
        ak4_is_matched_topb = ak.fill_none(
            ak4_topb_genjet_dr < self._ak4_genjet_topb_dr,
            False,
        )

        genjet_parton_flavour = (
            genjets_from_top_b.partonFlavour
            if "partonFlavour" in genjets_from_top_b.fields
            else ak.ones_like(genjets_from_top_b.pt) * PAD_VAL
        )
        genjet_hadron_flavour = (
            genjets_from_top_b.hadronFlavour
            if "hadronFlavour" in genjets_from_top_b.fields
            else ak.ones_like(genjets_from_top_b.pt) * PAD_VAL
        )
        matched_parton_flavour = (
            matched_topb_genjet.partonFlavour
            if "partonFlavour" in matched_topb_genjet.fields
            else ak.ones_like(matched_topb_genjet.pt) * PAD_VAL
        )
        matched_hadron_flavour = (
            matched_topb_genjet.hadronFlavour
            if "hadronFlavour" in matched_topb_genjet.fields
            else ak.ones_like(matched_topb_genjet.pt) * PAD_VAL
        )

        return {
            "nGenTopB": ak.num(gen_top_b).to_numpy(),
            "nGenJetFromTopB": ak.num(genjets_from_top_b).to_numpy(),
            "GenTopBPt": pad_val(gen_top_b.pt, self._num_gen_top_b, axis=1),
            "GenTopBEta": pad_val(gen_top_b.eta, self._num_gen_top_b, axis=1),
            "GenTopBPhi": pad_val(gen_top_b.phi, self._num_gen_top_b, axis=1),
            "GenTopBMass": pad_val(gen_top_b.mass, self._num_gen_top_b, axis=1),
            "GenTopBParentPdgId": pad_val(gen_top_b.topParentPdgId, self._num_gen_top_b, axis=1),
            "GenJetFromTopBPt": pad_val(genjets_from_top_b.pt, self._num_genjet_top_b, axis=1),
            "GenJetFromTopBEta": pad_val(genjets_from_top_b.eta, self._num_genjet_top_b, axis=1),
            "GenJetFromTopBPhi": pad_val(genjets_from_top_b.phi, self._num_genjet_top_b, axis=1),
            "GenJetFromTopBMass": pad_val(genjets_from_top_b.mass, self._num_genjet_top_b, axis=1),
            "GenJetFromTopBIdx": pad_val(
                genjets_from_top_b.genJetIdx, self._num_genjet_top_b, axis=1
            ),
            "GenJetFromTopBPartonFlavour": pad_val(
                genjet_parton_flavour, self._num_genjet_top_b, axis=1
            ),
            "GenJetFromTopBHadronFlavour": pad_val(
                genjet_hadron_flavour, self._num_genjet_top_b, axis=1
            ),
            "GenJetFromTopBMatchedGenBdr": pad_val(
                genjets_from_top_b.matchedGenBdr, self._num_genjet_top_b, axis=1
            ),
            "GenJetFromTopBParentPdgId": pad_val(
                genjets_from_top_b.topParentPdgId, self._num_genjet_top_b, axis=1
            ),
            "ak4JetIsMatchedToTopBGenJet": pad_val(
                ak4_is_matched_topb, num_ak4_jets, False, axis=1
            ).astype(int),
            "ak4JetMatchedTopBGenJetPt": pad_val(matched_topb_genjet.pt, num_ak4_jets, axis=1),
            "ak4JetMatchedTopBGenJetEta": pad_val(matched_topb_genjet.eta, num_ak4_jets, axis=1),
            "ak4JetMatchedTopBGenJetPhi": pad_val(matched_topb_genjet.phi, num_ak4_jets, axis=1),
            "ak4JetMatchedTopBGenJetMass": pad_val(matched_topb_genjet.mass, num_ak4_jets, axis=1),
            "ak4JetMatchedTopBGenJetIdx": pad_val(
                matched_topb_genjet.genJetIdx, num_ak4_jets, axis=1
            ),
            "ak4JetMatchedTopBGenJetDR": pad_val(ak4_topb_genjet_dr, num_ak4_jets, axis=1),
            "ak4JetMatchedTopBParentPdgId": pad_val(
                matched_topb_genjet.topParentPdgId, num_ak4_jets, axis=1
            ),
            "ak4JetMatchedTopBGenJetPartonFlavour": pad_val(
                matched_parton_flavour, num_ak4_jets, axis=1
            ),
            "ak4JetMatchedTopBGenJetHadronFlavour": pad_val(
                matched_hadron_flavour, num_ak4_jets, axis=1
            ),
        }

    def process(self, events: ak.Array):
        """Runs event processor for different types of jets"""

        start = time.time()
        self._chunk_counter += 1
        dataset_name = events.metadata.get("dataset", "unknown")
        events_factory = events.behavior.get("__events_factory__", None)
        partition_key = getattr(events_factory, "_partition_key", "unknown")
        entry_start = events.metadata.get("entrystart", "?")
        entry_stop = events.metadata.get("entrystop", "?")

        logging.info(
            "Chunk %s | dataset=%s | entries=[%s, %s) | nevents=%s | partition=%s",
            self._chunk_counter,
            dataset_name,
            entry_start,
            entry_stop,
            len(events),
            partition_key,
        )
        logging.info(f"Processing {dataset_name} with {len(events)} events")
        logging.info(f"# events {len(events)}")

        year = events.metadata["dataset"].split("_")[0]
        dataset = "_".join(events.metadata["dataset"].split("_")[1:])
        isData = not hasattr(events, "genWeight")

        # datasets for saving jec variations
        isJECs = (  # noqa: F841
            "HHto4B" in dataset
            or "TT" in dataset
            or "Wto2Q" in dataset
            or "Zto2Q" in dataset
            or "Hto2B" in dataset
            or "WW" in dataset
            or "ZZ" in dataset
            or "WZ" in dataset
        )

        # gen-weights
        gen_weights = events["genWeight"].to_numpy() if not isData else None
        n_events = len(events) if isData else np.sum(gen_weights)

        # selection and cutflow
        selection = PackedSelection()
        cutflow = OrderedDict()
        cutflow["all"] = n_events
        selection_args = (selection, cutflow, isData, gen_weights)

        # JEC factory loader
        JEC_loader = JECs(year)

        #########################
        # Object definitions
        #########################

        print("starting object selection", f"{time.time() - start:.2f}")

        # Leptons
        num_leptons = 2
        electrons, etrigvars = objects.good_electrons(
            events, events.Electron, **self.lepton_selection, year=year
        )
        muons, mtrigvars = objects.good_muons(
            events, events.Muon, **self.lepton_selection, year=year
        )
        taus, ttrigvars = objects.good_taus(events, events.Tau, year)
        boostedtaus = objects.good_boostedtaus(events, events.boostedTau)

        # SubJets
        num_subjets = 3
        subjets = events.SubJet

        # These are bools saying if the lepton is matched to a trigger object or not
        trigMatchVars = {**etrigvars, **mtrigvars, **ttrigvars}
        for key, val in trigMatchVars.items():
            trigMatchVars[key] = pad_val(val, num_leptons, False, axis=1).astype(int)

        print("Leptons", f"{time.time() - start:.2f}")

        # TODO: lepton systematics

        # AK4 Jets
        num_ak4_jets = 4
        jets, jec_shifted_jetvars = JEC_loader.get_jec_jets(
            events,
            events.Jet,
            year,
            isData,
            jecs=utils.jecs,
            fatjets=False,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print("events.Jet fields:", events.Jet.fields)
        print("jets fields after JEC:", jets.fields)

        if JEC_loader.met_factory is not None:
            met = JEC_loader.met_factory.build(events.MET, jets, {}) if isData else events.MET
        else:
            met = events.MET

        print("ak4 JECs", f"{time.time() - start:.2f}")

        jets = objects.good_ak4jets(jets, nano_version=self._nano_version)
        ht = ak.sum(jets.pt, axis=1)
        print("ak4", f"{time.time() - start:.2f}")

        # AK8 Jets
        num_ak8_jets = 3
        all_fatjets = objects.get_ak8jets(
            events.FatJet
        )  # this adds all our extra variables e.g. TXbb
        all_fatjets, jec_shifted_fatjetvars = JEC_loader.get_jec_jets(
            events,
            all_fatjets,
            year,
            isData,
            jecs=utils.jecs,
            fatjets=True,
            applyData=True,
            dataset=dataset,
            nano_version=self._nano_version,
        )
        print("ak8 JECs", f"{time.time() - start:.2f}")

        fatjets = objects.good_ak8jets(
            all_fatjets, **self.fatjet_selection, nano_version=self._nano_version
        )

        # VBF objects
        vbf_jets = objects.vbf_jets(
            jets,
            fatjets[:, :2],
            events,
            **self.vbf_jet_selection,
            **self.vbf_veto_lepton_selection,
        )

        # # AK4 objects away from first two fatjets AT THE SAME TIME
        ak4_jets_awayfromak8 = objects.ak4_jets_awayfromak8(
            jets,
            fatjets[:, :2],
            events,
            **self.ak4_bjet_selection,
            **self.ak4_bjet_lepton_selection,
            sort_by="nearest",
        )

        # # AK4 objects away from first fatjet only
        ak4_jets_awayfromFJ0 = objects.ak4_jets_awayfromFJ(
            jets,
            fatjets[:, :1],
            events,
            **self.ak4_bjet_selection,
            **self.ak4_bjet_lepton_selection,
        )

        # # AK4 objects away from second fatjet only
        ak4_jets_awayfromFJ1 = objects.ak4_jets_awayfromFJ(
            jets,
            fatjets[:, 1:2],
            events,
            **self.ak4_bjet_selection,
            **self.ak4_bjet_lepton_selection,
        )

        # # JMSR
        # # TODO: add variations per variable
        # bb_jmsr_shifted_vars = get_jmsr(
        #     fatjets_xbb,
        #     2,
        #     jmsr_vars=self.jmsr_vars,
        #     jms_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
        #     jmr_values={key: [1.0, 0.9, 1.1] for key in self.jmsr_vars},
        #     isData=isData,
        # )

        # fatjets = objects.get_CA_MASS(fatjets, boostedtaus, met, subjets, muons, electrons)
        fatjets = objects.get_CA_MASS(fatjets, taus, met, subjets, muons, electrons)
        print("CA mass", f"{time.time() - start:.2f}")

        #########################
        # Save / derive variables
        #########################

        # Gen variables - saving HH and bbbb 4-vector info
        genVars = {}
        for d in gen_selection_dict:
            if d in dataset:
                vars_dict = gen_selection_dict[d](events, fatjets, selection_args)
                genVars = {**genVars, **vars_dict}

        # TTbar truth information and AK4 -> top-b GenJet matching.
        # Filled with PAD_VAL/0 for data or samples without GenPart/GenJet.
        ttbarGenMatchVars = self.get_ttbar_gen_matching_vars(events, jets, num_ak4_jets)

        # used for normalization to cross section below
        gen_selected = (
            selection.all(*selection.names)
            if len(selection.names)
            else np.ones(len(events)).astype(bool)
        )
        logging.info(f"Passing gen selection: {np.sum(gen_selected)} / {len(events)}")

        # Lepton variables
        electronVars = {
            f"Electron{key}": pad_val(electrons[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Lepton"].items()
        }
        muonVars = {
            f"Muon{key}": pad_val(muons[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Lepton"].items()
        }
        tauVars = {
            f"Tau{key}": pad_val(taus[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["Tau"].items()
        }
        boostedtauVars = {
            f"BoostedTau{key}": pad_val(boostedtaus[var], num_leptons, axis=1)
            for (var, key) in self.skim_vars["BoostedTau"].items()
        }
        leptonVars = {**electronVars, **muonVars, **tauVars, **boostedtauVars}

        # Subjets
        subjetVars = {
            f"SubJet{key}": pad_val(subjets[var], num_subjets, axis=1)
            for (var, key) in self.skim_vars["SubJet"].items()
        }

        # AK4 Jet variables
        jet_skimvars = self.skim_vars["Jet"]
        if not isData:
            jet_skimvars = {
                **jet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak4JetVars = {
            f"ak4Jet{key}": pad_val(jets[var], num_ak4_jets, axis=1)
            for (var, key) in jet_skimvars.items()
        }

        if len(ak4_jets_awayfromak8) == 2:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(
                    ak.concatenate(
                        [ak4_jets_awayfromak8[0][var], ak4_jets_awayfromak8[1][var]], axis=1
                    ),
                    2,
                    axis=1,
                )
                for (var, key) in jet_skimvars.items()
            }
        else:
            ak4JetAwayVars = {
                f"AK4JetAway{key}": pad_val(ak4_jets_awayfromak8[var], 2, axis=1)
                for (var, key) in jet_skimvars.items()
            }

        if len(ak4_jets_awayfromFJ0) == 4:
            ak4JetAwayFJ0Vars = {
                f"AK4JetAwayFJ0{key}": pad_val(
                    ak.concatenate(
                        [ak4_jets_awayfromFJ0[0][var], ak4_jets_awayfromFJ0[1][var]], axis=1
                    ),
                    2,
                    axis=1,
                )
                for (var, key) in jet_skimvars.items()
            }
        else:
            ak4JetAwayFJ0Vars = {
                f"AK4JetAwayFJ0{key}": pad_val(ak4_jets_awayfromFJ0[var], 4, axis=1)
                for (var, key) in jet_skimvars.items()
            }

        if len(ak4_jets_awayfromFJ1) == 2:
            ak4JetAwayFJ1Vars = {
                f"AK4JetAwayFJ1{key}": pad_val(
                    ak.concatenate(
                        [ak4_jets_awayfromFJ1[0][var], ak4_jets_awayfromFJ1[1][var]], axis=1
                    ),
                    2,
                    axis=1,
                )
                for (var, key) in jet_skimvars.items()
            }
        else:
            ak4JetAwayFJ1Vars = {
                f"AK4JetAwayFJ1{key}": pad_val(ak4_jets_awayfromFJ1[var], 4, axis=1)
                for (var, key) in jet_skimvars.items()
            }

        # AK8 Jet variables
        fatjet_skimvars = self.skim_vars["FatJet"]
        if not isData:
            fatjet_skimvars = {
                **fatjet_skimvars,
                "pt_gen": "MatchedGenJetPt",
            }

        ak8FatJetVars = {
            f"ak8FatJet{key}": pad_val(fatjets[var], num_ak8_jets, axis=1)
            for (var, key) in fatjet_skimvars.items()
        }
        print("Jet vars", f"{time.time() - start:.2f}")

        # # JEC and JMSR
        # if self._region == "signal" and isJECs:
        #     # Jet JEC variables
        #     for var in ["pt"]:
        #         key = self.skim_vars["Jet"][var]
        #         for shift, vals in jec_shifted_jetvars[var].items():
        #             if shift != "":
        #                 ak4JetVars[f"ak4Jet{key}_{shift}"] = pad_val(vals, num_ak4_jets, axis=1)

        #     # FatJet JEC variables
        #     for var in ["pt"]:
        #         key = self.skim_vars["FatJet"][var]
        #         for shift, vals in jec_shifted_bbfatjetvars[var].items():
        #             if shift != "":
        #                 bbFatJetVars[f"bbFatJet{key}_{shift}"] = pad_val(vals, 2, axis=1)

        #     # FatJet JMSR
        #     for var in self.jmsr_vars:
        #         key = fatjet_skimvars[var]
        #         bbFatJetVars[f"bbFatJet{key}_raw"] = bbFatJetVars[f"bbFatJet{key}"]
        #         for shift, vals in bb_jmsr_shifted_vars[var].items():
        #             # overwrite saved mass vars with corrected ones
        #             label = "" if shift == "" else "_" + shift
        #             bbFatJetVars[f"bbFatJet{key}{label}"] = vals

        # MET
        metVars = {f"MET{key}": met[var].to_numpy() for (var, key) in self.skim_vars["MET"].items()}

        # Event variables
        eventVars = {
            key: events[val].to_numpy()
            for key, val in self.skim_vars["Event"].items()
            if key in events.fields
        }
        eventVars["ht"] = ht.to_numpy()
        eventVars["nElectrons"] = ak.num(electrons).to_numpy()
        eventVars["nMuons"] = ak.num(muons).to_numpy()
        eventVars["nTaus"] = ak.num(taus).to_numpy()
        eventVars["nBoostedTaus"] = ak.num(boostedtaus).to_numpy()
        eventVars["nJets"] = ak.num(jets).to_numpy()
        eventVars["nFatJets"] = ak.num(fatjets).to_numpy()
        eventVars["nSubJets"] = ak.num(subjets).to_numpy()

        # jin for CA
        # eventVars["CA_matched_tau_pt_sum"] = ca_tau_pt_sum.to_numpy()
        # eventVars["CA_tau_idx_0"] = ca_tau_indices[:, 0].to_numpy()
        # eventVars["CA_tau_idx_1"] = ca_tau_indices[:, 1].to_numpy()
        # eventVars["CA_best_fatjet_idx"] = ca_best_fatjet_idx.to_numpy()

        if isData:
            pileupVars = {key: np.ones(len(events)) * PAD_VAL for key in self.skim_vars["Pileup"]}
        else:
            pileupVars = {key: events.Pileup[key].to_numpy() for key in self.skim_vars["Pileup"]}
        pileupVars = {**pileupVars, "nPV": events.PV["npvs"].to_numpy()}

        # Trigger variables
        HLTVars = {}
        zeros = np.zeros(len(events), dtype="int")
        for trigger in self.HLTs[year]:
            if trigger in events.HLT.fields:
                HLTVars[f"HLT_{trigger}"] = events.HLT[trigger].to_numpy().astype(int)
            else:
                logger.warning(f"Missing {trigger}!")
                HLTVars[f"HLT_{trigger}"] = zeros

        print("HLT vars", f"{time.time() - start:.2f}")

        # vbfJets
        vbfJetVars = {
            f"VBFJet{key}": pad_val(vbf_jets[var], 2, axis=1)
            for (var, key) in self.skim_vars["Jet"].items()
        }

        # # JEC variations for VBF Jets
        # if self._region == "signal" and isJECs:
        #     for var in ["pt"]:
        #         key = self.skim_vars["Jet"][var]
        #         for label, shift in utils.jecs.items():
        #             if shift in ak.fields(vbf_jets):
        #                 for vari in ["up", "down"]:
        #                     vbfJetVars[f"VBFJet{key}_{label}_{vari}"] = pad_val(
        #                         vbf_jets[shift][vari][var], 2, axis=1
        #                     )

        skimmed_events = {
            **genVars,
            **ttbarGenMatchVars,
            **eventVars,
            **pileupVars,
            **trigMatchVars,
            **HLTVars,
            **ak4JetAwayVars,
            **ak4JetAwayFJ0Vars,
            **ak4JetAwayFJ1Vars,
            **leptonVars,
            **ak4JetVars,
            **ak8FatJetVars,
            **metVars,
            **subjetVars,
            # **bbFatJetVars,
            # **trigObjFatJetVars,
            **vbfJetVars,
        }

        # if self._region == "signal":
        #     bdtVars = self.getBDT(bbFatJetVars, vbfJetVars, ak4JetAwayVars, met_pt, "")
        #     print(bdtVars)
        #     skimmed_events = {
        #         **skimmed_events,
        #         **bdtVars,
        #     }

        print("Vars", f"{time.time() - start:.2f}")

        ######################
        # Selection
        ######################

        HLT_triggered = np.any(
            np.array(
                [events.HLT[trigger] for trigger in self.HLTs[year] if trigger in events.HLT.fields]
            ),
            axis=0,
        )

        # don't apply triggers for now, for trigger studies etc.
        apply_trigger = True
        if apply_trigger:
            add_selection("trigger", HLT_triggered, *selection_args)

        # metfilters
        cut_metfilters = np.ones(len(events), dtype="bool")
        for mf in utils.met_filters:
            if mf in events.Flag.fields:
                cut_metfilters = cut_metfilters & events.Flag[mf]
        add_selection("met_filters", cut_metfilters, *selection_args)

        # jet veto maps
        cut_jetveto = get_jetveto_event(jets, year)
        add_selection("ak4_jetveto", cut_jetveto, *selection_args)

        # 1 Electron or 1 Muon passing selection
        add_selection(
            "singleLepton", (ak.num(electrons) == 1) | (ak.num(muons) == 1), *selection_args
        )

        # >=1 AK8 jets with mass cut (230 GeV by default)
        if self.fatjet_selection["mass"] >= 0:  # if < 0, don't apply any fatjet selection
            cut_mass = (
                np.sum(ak8FatJetVars["ak8FatJetMass"] >= self.fatjet_selection["mass"], axis=1) >= 1
            )
            add_selection("ak8_mass", cut_mass, *selection_args)

        # # 1 AK8 jets passing primary selection, and no jets passing only secondary selection
        # add_selection(
        #     "ak8_singleFatjet",
        #     (ak.num(fatjets) == 1),  # & (ak.num(secondary_fatjets) == 1),
        #     *selection_args,
        # )

        # # >=1 AK8 jets with pT cut (230 GeV by default)
        # if self.fatjet_selection["pt"] >= 0:  # if < 0, don't apply any fatjet selection
        #     cut_pt = (
        #         np.sum(ak8FatJetVars["ak8FatJetPt"] >= self.fatjet_selection["pt"], axis=1) >= 1
        #     )
        #     add_selection("ak8_pt", cut_pt, *selection_args)

        # # >=1 AK8 jets with mSD >= 40 GeV
        # cut_mass = np.sum(ak8FatJetVars["ak8FatJetMsd"] >= 40, axis=1) >= 1
        # add_selection("ak8_mass", cut_mass, *selection_args)

        # Veto leptons
        # add_selection(
        #     "0lep",
        #     (ak.sum(veto_muon_sel, axis=1) == 0) & (ak.sum(veto_electron_sel, axis=1) == 0),
        #     *selection_args,
        # )

        # if self._region == "signal":
        #     # >=1 bb AK8 jets (ordered by TXbb) with TXbb > 0.8
        #     cut_txbb = (
        #         np.sum(
        #             bbFatJetVars[f"bbFatJet{txbb_str}"] >= self.preselection[self.txbb],
        #             axis=1,
        #         )
        #         >= 1
        #     )
        #     add_selection("ak8bb_txbb0", cut_txbb, *selection_args)

        # VBF veto cut (not now)
        # add_selection("vbf_veto", ~(cut_vbf), *selection_args)

        if self._fatjet_bb_preselection:
            # at least 1 jet with ParTXbbvsQCDTop > 0.3
            cut_bb = (
                np.sum(
                    ak8FatJetVars["ak8FatJetParTXbbvsQCDTop"] >= self.preselection["glopart-v2"],
                    axis=1,
                )
                >= 1
            )
            add_selection("ak8_bb_preselection", cut_bb, *selection_args)

        if self._prescale_factor:
            cut_prescale = events.event % self._prescale_factor == 0
            add_selection("prescale", cut_prescale, *selection_args)

        print("Selection", f"{time.time() - start:.2f}")

        ######################
        # Weights
        ######################

        totals_dict = {"nevents": n_events}

        if isData:
            skimmed_events["weight"] = np.ones(n_events)
        else:
            weights_dict, totals_temp = self.add_weights(
                events,
                year,
                dataset,
                gen_weights,
                gen_selected,
            )
            skimmed_events = {**skimmed_events, **weights_dict}
            totals_dict = {**totals_dict, **totals_temp}

        ##############################
        # Reshape and apply selections
        ##############################

        sel_all = selection.all(*selection.names)
        skimmed_events = {
            key: value.reshape(len(skimmed_events["weight"]), -1)[sel_all]
            for (key, value) in skimmed_events.items()
        }

        dataframe = self.to_pandas(skimmed_events)
        fname = events.behavior["__events_factory__"]._partition_key.replace("/", "_") + ".parquet"
        self.dump_table(dataframe, fname)

        logger.info(f"Cutflow:\n{cutflow}")

        print("Return ", f"{time.time() - start:.2f}")
        print("Columns:", print(list(dataframe.columns)))
        return {year: {dataset: {"totals": totals_dict, "cutflow": cutflow}}}

    def postprocess(self, accumulator):
        return accumulator

    def add_weights(
        self,
        events,
        year,
        dataset,
        gen_weights,
        gen_selected,
    ) -> tuple[dict, dict]:
        """Adds weights and variations, saves totals for all norm preserving weights and variations"""
        weights = Weights(len(events), storeIndividual=True)
        weights.add("genweight", gen_weights)

        add_pileup_weight(weights, year, events.Pileup.nPU.to_numpy(), dataset)
        add_ps_weight(weights, events.PSWeight)

        logger.debug("weights", extra=weights._weights.keys())

        ###################### Save all the weights and variations ######################

        # these weights should not change the overall normalization, so are saved separately
        norm_preserving_weights = hh_vars.norm_preserving_weights

        # dictionary of all weights and variations
        weights_dict = {}
        # dictionary of total # events for norm preserving variations for normalization in postprocessing
        totals_dict = {}

        # nominal
        weights_dict["weight"] = weights.weight()

        # norm preserving weights, used to do normalization in post-processing
        weight_np = weights.partial_weight(include=norm_preserving_weights)
        totals_dict["np_nominal"] = np.sum(weight_np[gen_selected])

        if self._systematics:
            for systematic in list(weights.variations):
                weights_dict[f"weight_{systematic}"] = weights.weight(modifier=systematic)

                if utils.remove_variation_suffix(systematic) in norm_preserving_weights:
                    var_weight = weights.partial_weight(include=norm_preserving_weights)
                    # modify manually
                    if "Down" in systematic and systematic not in weights._modifiers:
                        var_weight = (
                            var_weight / weights._modifiers[systematic.replace("Down", "Up")]
                        )
                    else:
                        var_weight = var_weight * weights._modifiers[systematic]

                    # need to save total # events for each variation for normalization in post-processing
                    totals_dict[f"np_{systematic}"] = np.sum(var_weight[gen_selected])

        # TEMP: save each individual weight TODO: remove
        for key in weights._weights:
            weights_dict[f"single_weight_{key}"] = weights.partial_weight([key])

        ###################### alpha_S and PDF variations ######################

        if ("HHTobbbb" in dataset or "HHto4B" in dataset) or dataset.startswith("TTTo"):
            scale_weights = get_scale_weights(events)
            if scale_weights is not None:
                weights_dict["scale_weights"] = (
                    scale_weights * weights_dict["weight"][:, np.newaxis]
                )
                totals_dict["np_scale_weights"] = np.sum(
                    (scale_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
                )

        if "HHTobbbb" in dataset or "HHto4B" in dataset:
            pdf_weights = get_pdf_weights(events)
            weights_dict["pdf_weights"] = pdf_weights * weights_dict["weight"][:, np.newaxis]
            totals_dict["np_pdf_weights"] = np.sum(
                (pdf_weights * weight_np[:, np.newaxis])[gen_selected], axis=0
            )

        ###################### Normalization (Step 1) ######################

        weight_norm = self.get_dataset_norm(year, dataset)
        # print("year", year, "dataset", dataset, "weight norm", weight_norm)
        # normalize all the weights to xsec, needs to be divided by totals in Step 2 in post-processing
        for key, val in weights_dict.items():
            weights_dict[key] = val * weight_norm

        # save the unnormalized weight, to confirm that it's been normalized in post-processing
        weights_dict["weight_noxsec"] = weights.weight()

        return weights_dict, totals_dict
