#!/bin/bash

# ROS2命令启动脚本 - 适用于Ubuntu 22.04
# 作者: 用户自定义脚本
# 版本: 1.2 - 添加了follow命令和tab补全功能

# 全局变量
CURRENT_PID=""
SCRIPT_RUNNING=true

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 可用命令列表
AVAILABLE_COMMANDS=("torque" "inference" "follow" "pico" "help" "exit")

# 补全函数
_ros2_completions() {
    local cur="${COMP_WORDS[COMP_CWORD]}"
    COMPREPLY=($(compgen -W "${AVAILABLE_COMMANDS[*]}" -- "$cur"))
}

# 信号处理函数
cleanup() {
    echo -e "\n${YELLOW}收到中断信号...${NC}"
    
    if [ -n "$CURRENT_PID" ] && kill -0 $CURRENT_PID 2>/dev/null; then
        echo -e "${YELLOW}正在停止当前运行的程序 (PID: $CURRENT_PID)...${NC}"
        kill -TERM $CURRENT_PID 2>/dev/null
        wait $CURRENT_PID 2>/dev/null
        CURRENT_PID=""
        echo -e "${GREEN}程序已停止${NC}"
        echo -e "${BLUE}脚本继续运行，再次按 Ctrl+C 退出脚本${NC}"
    else
        echo -e "${GREEN}正在退出脚本...${NC}"
        SCRIPT_RUNNING=false
        exit 0
    fi
}

# 设置信号陷阱
trap cleanup SIGINT

# 检查ROS2环境
check_ros2_env() {
    if [ -z "$ROS_DISTRO" ]; then
        echo -e "${RED}错误: 未检测到ROS2环境，请先source您的ROS2设置${NC}"
        echo -e "${YELLOW}例如: source /opt/ros/humble/setup.bash${NC}"
        return 1
    fi
    echo -e "${GREEN}ROS2环境检测正常 (发行版: $ROS_DISTRO)${NC}"
    return 0
}

# 执行torque命令
execute_torque() {
    echo -e "${BLUE}启动 Open Manipulator X Hardware...${NC}"
    echo -e "${YELLOW}执行命令: ros2 launch open_manipulator_x_bringup hardware.launch.py${NC}"
    echo "----------------------------------------"
    
    ros2 launch open_manipulator_x_bringup hardware.launch.py &
    CURRENT_PID=$!
    
    # 等待程序完成或被中断
    wait $CURRENT_PID 2>/dev/null
    local exit_code=$?
    CURRENT_PID=""
    
    echo "----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Torque程序正常结束${NC}"
    else
        echo -e "${YELLOW}Torque程序被中断或异常退出${NC}"
    fi
}

# 执行inference命令
execute_inference() {
    echo -e "${BLUE}启动 Leapsim Inference...${NC}"
    echo -e "${YELLOW}执行命令: ros2 run leapsim inference${NC}"
    echo "----------------------------------------"
    
    ros2 run leapsim inference &
    CURRENT_PID=$!
    
    # 等待程序完成或被中断
    wait $CURRENT_PID 2>/dev/null
    local exit_code=$?
    CURRENT_PID=""
    
    echo "----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Inference程序正常结束${NC}"
    else
        echo -e "${YELLOW}Inference程序被中断或异常退出${NC}"
    fi
}

# 执行follow命令
execute_follow() {
    echo -e "${BLUE}启动 Follow Follow...${NC}"
    echo -e "${YELLOW}执行命令: ros2 run follow follow${NC}"
    echo "----------------------------------------"
    
    ros2 run follow follow &
    CURRENT_PID=$!
    
    # 等待程序完成或被中断
    wait $CURRENT_PID 2>/dev/null
    local exit_code=$?
    CURRENT_PID=""
    
    echo "----------------------------------------"
    if [ $exit_code -eq 0 ]; then
        echo -e "${GREEN}Follow程序正常结束${NC}"
    else
        echo -e "${YELLOW}Follow程序被中断或异常退出${NC}"
    fi
}

# 执行pico命令
execute_pico() {
    echo -e "${BLUE}启动 Pico Node...${NC}"
    echo -e "${YELLOW}执行命令: ros2 run pico pico_node${NC}"
    echo -e "${YELLOW}将在5秒后询问是否继续执行follow命令${NC}"
    echo "----------------------------------------"
    
    # 启动pico_node并捕获输出
    ros2 run pico pico_node &
    CURRENT_PID=$!
    
    # 前5秒显示输出
    local count=0
    while [ $count -lt 5 ] && kill -0 $CURRENT_PID 2>/dev/null; do
        sleep 1
        count=$((count + 1))
        echo -e "${BLUE}Pico运行中... ($count/5秒)${NC}"
    done
    
    # 检查程序是否还在运行
    if ! kill -0 $CURRENT_PID 2>/dev/null; then
        echo -e "${RED}Pico程序在5秒内异常退出${NC}"
        CURRENT_PID=""
        return 1
    fi
    
    echo "----------------------------------------"
    echo -e "${YELLOW}Pico程序继续在后台运行...${NC}"
    
    # 询问是否运行follow命令
    while true; do
        echo -e "${BLUE}是否要运行 'ros2 run follow pico' 命令？(yes/no):${NC}"
        read -r response
        
        case "$response" in
            yes|y|Y|YES)
                echo -e "${GREEN}启动 Follow Pico...${NC}"
                echo -e "${YELLOW}执行命令: ros2 run follow pico${NC}"
                echo "----------------------------------------"
                
                # 启动follow命令
                ros2 run follow pico &
                local follow_pid=$!
                
                # 等待任一程序结束
                wait $follow_pid 2>/dev/null
                local follow_exit=$?
                
                # 清理pico进程
                if kill -0 $CURRENT_PID 2>/dev/null; then
                    kill -TERM $CURRENT_PID 2>/dev/null
                    wait $CURRENT_PID 2>/dev/null
                fi
                CURRENT_PID=""
                
                echo "----------------------------------------"
                if [ $follow_exit -eq 0 ]; then
                    echo -e "${GREEN}Follow程序正常结束${NC}"
                else
                    echo -e "${YELLOW}Follow程序被中断或异常退出${NC}"
                fi
                break
                ;;
            no|n|N|NO)
                echo -e "${YELLOW}停止Pico程序...${NC}"
                if kill -0 $CURRENT_PID 2>/dev/null; then
                    kill -TERM $CURRENT_PID 2>/dev/null
                    wait $CURRENT_PID 2>/dev/null
                    echo -e "${GREEN}Pico程序已停止${NC}"
                else
                    echo -e "${YELLOW}Pico程序已经停止${NC}"
                fi
                CURRENT_PID=""
                break
                ;;
            *)
                echo -e "${RED}无效输入，请输入 yes 或 no${NC}"
                ;;
        esac
    done
}

# 显示帮助信息
show_help() {
    echo -e "${BLUE}=== ROS2命令启动脚本帮助 ===${NC}"
    echo -e "${GREEN}可用命令:${NC}"
    echo -e "  ${YELLOW}torque${NC}    - 启动 Open Manipulator X Hardware"
    echo -e "  ${YELLOW}inference${NC} - 启动 Leapsim Inference"
    echo -e "  ${YELLOW}follow${NC}    - 启动 Follow Follow"
    echo -e "  ${YELLOW}pico${NC}      - 启动 Pico Node (5秒后询问是否运行follow)"
    echo -e "  ${YELLOW}help${NC}      - 显示此帮助信息"
    echo -e "  ${YELLOW}exit${NC}      - 退出脚本"
    echo ""
    echo -e "${BLUE}使用说明:${NC}"
    echo -e "- 支持命令前缀匹配：${YELLOW}t${NC} → torque, ${YELLOW}inf${NC} → inference, ${YELLOW}f${NC} → follow"
    echo -e "- 使用 ${YELLOW}Tab键${NC} 进行命令自动补全 (如果支持)"
    echo -e "- 使用 ${YELLOW}Ctrl+C${NC} 停止当前运行的程序"
    echo -e "- 当无程序运行时，再次使用 ${YELLOW}Ctrl+C${NC} 退出脚本"
    echo -e "- 确保已经source了ROS2环境设置"
    echo "=================================="
}

# 智能命令匹配
match_command() {
    local input="$1"
    local matches=()
    
    # 精确匹配优先
    for cmd in "${AVAILABLE_COMMANDS[@]}"; do
        if [ "$cmd" = "$input" ]; then
            echo "$cmd"
            return 0
        fi
    done
    
    # 前缀匹配
    for cmd in "${AVAILABLE_COMMANDS[@]}"; do
        if [[ "$cmd" == "$input"* ]]; then
            matches+=("$cmd")
        fi
    done
    
    case ${#matches[@]} in
        0)
            echo ""
            return 1
            ;;
        1)
            echo "${matches[0]}"
            return 0
            ;;
        *)
            echo -e "${YELLOW}多个匹配项: ${matches[*]}${NC}"
            echo -e "${BLUE}请输入更多字符以区分命令${NC}"
            return 2
            ;;
    esac
}

# 启用Tab补全
enable_completion() {
    # 尝试启用补全功能
    if command -v complete >/dev/null 2>&1; then
        complete -F _ros2_completions ros2_script 2>/dev/null
        # 设置readline选项以支持Tab补全
        set completion-ignore-case on 2>/dev/null
        echo -e "${GREEN}Tab补全功能已启用${NC}"
        return 0
    else
        echo -e "${YELLOW}Tab补全功能不可用，但支持前缀匹配${NC}"
        return 1
    fi
}

# 主函数
main() {
    echo -e "${GREEN}=== ROS2命令启动脚本 ===${NC}"
    echo -e "${BLUE}Ubuntu 22.04 兼容版本 (智能命令匹配)${NC}"
    echo ""
    
    # 启用补全功能
    enable_completion
    echo ""
    
    # 检查ROS2环境
    if ! check_ros2_env; then
        exit 1
    fi
    
    show_help
    echo ""
    
    # 主循环
    while $SCRIPT_RUNNING; do
        echo -e "${BLUE}请输入命令:${NC}"
        
        # 使用简单的read命令，支持readline特性
        if command -v complete >/dev/null 2>&1; then
            read -e -p "> " command
        else
            read -p "> " command
        fi
        
        # 如果输入为空，继续循环
        if [ -z "$command" ]; then
            continue
        fi
        
        # 尝试智能匹配命令
        matched_cmd=$(match_command "$command")
        match_result=$?
        
        if [ $match_result -eq 0 ]; then
            if [ "$matched_cmd" != "$command" ]; then
                echo -e "${GREEN}匹配到命令: $matched_cmd${NC}"
            fi
            command="$matched_cmd"
        elif [ $match_result -eq 2 ]; then
            # 多个匹配项，继续下一轮输入
            echo ""
            continue
        fi
        
        case "$command" in
            torque)
                execute_torque
                ;;
            inference)
                execute_inference
                ;;
            follow)
                execute_follow
                ;;
            pico)
                execute_pico
                ;;
            help)
                show_help
                ;;
            exit)
                echo -e "${GREEN}正在退出脚本...${NC}"
                break
                ;;
            *)
                echo -e "${RED}未知命令: $command${NC}"
                echo -e "${YELLOW}输入 'help' 查看可用命令${NC}"
                ;;
        esac
        
        echo ""
    done
    
    # 脚本退出前清理
    if [ -n "$CURRENT_PID" ] && kill -0 $CURRENT_PID 2>/dev/null; then
        echo -e "${YELLOW}清理后台进程...${NC}"
        kill -TERM $CURRENT_PID 2>/dev/null
        wait $CURRENT_PID 2>/dev/null
    fi
    
    echo -e "${GREEN}脚本已退出${NC}"
}

# 运行主函数
main