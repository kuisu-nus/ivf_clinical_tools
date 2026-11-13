"""Oocyte-yield prediction: data pipeline, group model, similar-case retrieval and case database.

The package lives under task/services so the tool service finds it inside a sandbox
(task/services is mounted at /tmp/omni_services). What the service imports at run time
(ovu.ml.features, ovu.ml.model, ovu.ml.predict, ovu.similar.index, ovu.db.database, ovu.inputs)
needs numpy only; pandas and pyyaml are used by the offline steps (pipeline, training, builds).
"""
