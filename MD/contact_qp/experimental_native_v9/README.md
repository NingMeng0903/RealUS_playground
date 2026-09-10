Experimental native protocol v9 withdrawn from the active tree.

stage3_only.patch applies to HEAD plus the preserved Stage 1 loop changes; source/ contains exact pre-withdrawal files, including the new tests. tracked_full_HEAD.patch additionally includes the Stage 1 loop delta for archival completeness. Source hashes and the baseline commit are in manifest.json.

Last build succeeded (cmake --build rm75_control/native/wbc_rt/build -j2). Last completed suite predates the explicit-confirmation fixes; no final Stage 3 PASS was issued. Latest fixes were built but not regression tested before the user pause. Confirmation locking and explicit commit return propagation remain unfinished. The 96-active-row stress failed realtime acceptance and must not be interpreted as deployable. All runs used private SHM, with no hardware attachment. The prior performance/tests logs are historical evidence, not acceptance of the final paused source.
