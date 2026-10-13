# 离开本环境后，恢复激活前的 ROS 2、CUDA 和用户包配置。
if [ "${_GO2_RL_PYTHONPATH_SET-}" = x ]; then
    export PYTHONPATH="${_GO2_RL_PYTHONPATH-}"
else
    unset PYTHONPATH
fi
if [ "${_GO2_RL_LIBRARY_PATH_SET-}" = x ]; then
    export LD_LIBRARY_PATH="${_GO2_RL_LIBRARY_PATH-}"
else
    unset LD_LIBRARY_PATH
fi
if [ "${_GO2_RL_USER_SITE_SET-}" = x ]; then
    export PYTHONNOUSERSITE="${_GO2_RL_USER_SITE-}"
else
    unset PYTHONNOUSERSITE
fi
unset _GO2_RL_PYTHONPATH_SET _GO2_RL_PYTHONPATH
unset _GO2_RL_LIBRARY_PATH_SET _GO2_RL_LIBRARY_PATH
unset _GO2_RL_USER_SITE_SET _GO2_RL_USER_SITE
