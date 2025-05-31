"""
A* Algoritması Real-time Görselleştirme
Algoritmanın çalışırken adım adım görünmesi
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
import time
import threading
from queue import Queue


class RealtimeAStarVisualizer:
    """Real-time A* görselleştirici"""
    
    def __init__(self, graph, astar_algorithm):
        self.graph = graph
        self.astar = astar_algorithm
        self.fig = None
        self.ax = None
        self.grid_display = None
        self.stats_text = None
        
        # Real-time için
        self.update_queue = Queue()
        self.is_running = False
        self.pause_between_steps = 0.1  # Saniye
        
        # Renkler - daha canlı tonlar
        self.colors = {
            'empty': (0.95, 0.95, 0.95, 1.0),    # Açık gri
            'wall': (0.1, 0.1, 0.1, 1.0),        # Siyah
            'start': (0.0, 0.8, 0.0, 1.0),       # Parlak yeşil
            'goal': (0.8, 0.0, 0.0, 1.0),        # Parlak kırmızı
            'path': (0.0, 0.0, 1.0, 1.0),        # Mavi
            'explored': (1.0, 1.0, 0.0, 0.7),    # Sarı (keşfedilen)
            'frontier': (1.0, 0.5, 0.0, 0.8),    # Turuncu (open set)
            'current': (1.0, 0.0, 1.0, 1.0),     # Magenta (şu anki)
            'considering': (0.0, 1.0, 1.0, 0.6)  # Cyan (değerlendirilen)
        }
        
        self.color_map = ListedColormap([
            self.colors['empty'],
            self.colors['wall'], 
            self.colors['start'],
            self.colors['goal'],
            self.colors['explored'],
            self.colors['frontier'],
            self.colors['current'],
            self.colors['path'],
            self.colors['considering']
        ])
        
        self.color_values = {
            'empty': 0, 'wall': 1, 'start': 2, 'goal': 3,
            'explored': 4, 'frontier': 5, 'current': 6, 
            'path': 7, 'considering': 8
        }
    
    def setup_interactive_plot(self):
        """İnteraktif plot kurulumu"""
        plt.ion()  # Interactive mode ON
        self.fig, self.ax = plt.subplots(figsize=(14, 10))
        
        # İlk grid
        grid = self.create_grid_array()
        self.grid_display = self.ax.imshow(
            grid, cmap=self.color_map, interpolation='nearest',
            vmin=0, vmax=len(self.color_values)-1
        )
        
        # Eksenleri ayarla
        self.ax.set_xlim(-0.5, self.graph.width - 0.5)
        self.ax.set_ylim(-0.5, self.graph.height - 0.5)
        self.ax.set_aspect('equal')
        
        # Grid çizgileri - daha ince
        self.ax.set_xticks(np.arange(-0.5, self.graph.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.graph.height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.3, alpha=0.4)
        
        # Başlık
        self.ax.set_title(f'A* Real-time Pathfinding - {self.astar.heuristic_name}', 
                         fontsize=16, fontweight='bold', pad=20)
        
        # İstatistik kutusu
        self.stats_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        )
        
        # Legend
        self.create_legend()
        
        # Kontrollar
        control_text = """
KONTROLLER:
• SPACE: Duraklat/Devam
• ENTER: Tek adım ilerle
• ESC: Durdur
• +/-: Hızı ayarla
        """
        
        self.ax.text(
            0.02, 0.02, control_text, transform=self.ax.transAxes,
            fontsize=9, verticalalignment='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.8)
        )
        
        plt.tight_layout()
        return self.fig, self.ax
    
    def create_grid_array(self):
        """Grid array oluştur"""
        grid = np.zeros((self.graph.height, self.graph.width))
        
        for x in range(self.graph.width):
            for y in range(self.graph.height):
                node = self.graph.get_node(x, y)
                if node:
                    if node.node_type == 'wall':
                        grid[y, x] = self.color_values['wall']
                    elif node.node_type == 'start':
                        grid[y, x] = self.color_values['start']
                    elif node.node_type == 'goal':
                        grid[y, x] = self.color_values['goal']
                    else:
                        grid[y, x] = self.color_values['empty']
        
        return grid
    
    def update_grid_realtime(self, step_info, path=None):
        """Grid'i real-time güncelle"""
        grid = self.create_grid_array()
        
        if step_info:
            # Closed set (keşfedilen)
            if 'closed_set' in step_info:
                for node in step_info['closed_set']:
                    if node.node_type == 'empty':
                        grid[node.y, node.x] = self.color_values['explored']
            
            # Open set (frontier)
            if 'open_set' in step_info:
                for node in step_info['open_set']:
                    if node.node_type == 'empty':
                        grid[node.y, node.x] = self.color_values['frontier']
            
            # Considering neighbors (anlık olarak değerlendirilen)
            if 'neighbors' in step_info:
                for neighbor in step_info['neighbors']:
                    if neighbor.node_type == 'empty' and not neighbor.visited:
                        grid[neighbor.y, neighbor.x] = self.color_values['considering']
            
            # Current node (şu anki)
            if 'current' in step_info:
                current = step_info['current']
                if current.node_type == 'empty':
                    grid[current.y, current.x] = self.color_values['current']
        
        # Path (bulunan yol)
        if path:
            for node in path:
                if node.node_type == 'empty':
                    grid[node.y, node.x] = self.color_values['path']
        
        # Start ve goal'ı her zaman göster
        if self.graph.start_node:
            start = self.graph.start_node
            grid[start.y, start.x] = self.color_values['start']
        
        if self.graph.goal_node:
            goal = self.graph.goal_node
            grid[goal.y, goal.x] = self.color_values['goal']
        
        return grid
    
    def update_stats_display(self, step_info, is_finished=False, final_stats=None):
        """İstatistik display'ini güncelle"""
        if is_finished and final_stats:
            stats_text = f"""
🎯 TAMAMLANDI!
Yol Bulundu: {"✅ Evet" if final_stats["path_found"] else "❌ Hayır"}
Keşfedilen Düğüm: {final_stats["nodes_explored"]}
Yol Uzunluğu: {final_stats["path_length"]}
Toplam Adım: {final_stats["total_steps"]}
Heuristic: {final_stats["heuristic_used"]}
            """
        elif step_info:
            current = step_info['current']
            stats_text = f"""
🔄 Çalışıyor... Adım: {step_info["step"]}
Mevcut Düğüm: ({current.x}, {current.y})
G-cost: {current.g_cost:.1f}
H-cost: {current.h_cost:.1f}
F-cost: {current.f_cost:.1f}

Open Set: {len(step_info.get("open_set", []))} düğüm
Closed Set: {len(step_info.get("closed_set", []))} düğüm
İşlem: {step_info.get("action", "Keşfediyor")}
            """
        else:
            stats_text = "🚀 Başlatılıyor..."
        
        self.stats_text.set_text(stats_text)
    
    def create_legend(self):
        """Legend oluştur"""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor=self.colors['empty'], label='🔲 Boş Alan'),
            Patch(facecolor=self.colors['wall'], label='🧱 Duvar'),
            Patch(facecolor=self.colors['start'], label='🟢 Başlangıç'),
            Patch(facecolor=self.colors['goal'], label='🔴 Hedef'),
            Patch(facecolor=self.colors['explored'], label='🟡 Keşfedilen'),
            Patch(facecolor=self.colors['frontier'], label='🟠 Sınır (Open)'),
            Patch(facecolor=self.colors['considering'], label='🔵 Değerlendirilen'),
            Patch(facecolor=self.colors['current'], label='🟣 Mevcut'),
            Patch(facecolor=self.colors['path'], label='🛣️ Bulunan Yol')
        ]
        
        self.ax.legend(handles=legend_elements, loc='center left', 
                      bbox_to_anchor=(1, 0.5), fontsize=10)
    
    def run_realtime_visualization(self, step_delay=0.1, interactive_mode=False):
        """Real-time görselleştirme çalıştır"""
        self.setup_interactive_plot()
        self.pause_between_steps = step_delay
        
        print(f"\n🎬 Real-time A* görselleştirmesi başlatılıyor...")
        print(f"⏱️ Adım gecikme süresi: {step_delay} saniye")
        if interactive_mode:
            print("🎮 İnteraktif mod: ENTER ile adım adım ilerleyin")
        
        # Algoritma adımlarını hesapla
        path, success, final_stats = self.astar.find_path(step_by_step=True)
        algorithm_steps = self.astar.algorithm_steps
        
        if not algorithm_steps:
            print("❌ Algoritma adımları bulunamadı!")
            return None
        
        print(f"📊 Toplam {len(algorithm_steps)} adım bulundu")
        
        try:
            # Her adımı göster
            for i, step_info in enumerate(algorithm_steps):
                # Grid'i güncelle
                grid = self.update_grid_realtime(step_info)
                self.grid_display.set_array(grid)
                
                # İstatistikleri güncelle
                self.update_stats_display(step_info)
                
                # Başlığı güncelle
                progress = (i + 1) / len(algorithm_steps) * 100
                self.ax.set_title(
                    f'A* Real-time - {self.astar.heuristic_name} | '
                    f'Adım {i+1}/{len(algorithm_steps)} (%{progress:.1f})',
                    fontsize=16, fontweight='bold'
                )
                
                # Çiz ve bekle
                plt.draw()
                plt.pause(0.001)  # Minimal pause for drawing
                
                if interactive_mode:
                    input(f"Adım {i+1} - Devam etmek için ENTER'a basın...")
                else:
                    time.sleep(self.pause_between_steps)
            
            # Son durumu göster - bulunan yol
            if success:
                print("✅ Yol bulundu! Son durumu gösteriliyor...")
                final_grid = self.update_grid_realtime(None, path)
                self.grid_display.set_array(final_grid)
                self.update_stats_display(None, True, final_stats)
                
                self.ax.set_title(
                    f'A* Tamamlandı - Yol Bulundu! | {self.astar.heuristic_name}',
                    fontsize=16, fontweight='bold', color='green'
                )
            else:
                print("❌ Yol bulunamadı!")
                self.ax.set_title(
                    f'A* Tamamlandı - Yol Bulunamadı | {self.astar.heuristic_name}',
                    fontsize=16, fontweight='bold', color='red'
                )
            
            plt.draw()
            
            # Final display
            print(f"\n📊 Final Sonuçlar:")
            print(f"   • Başarı: {'Evet' if success else 'Hayır'}")
            print(f"   • Keşfedilen düğüm: {final_stats['nodes_explored']}")
            print(f"   • Yol uzunluğu: {final_stats['path_length']}")
            print(f"   • Toplam adım: {len(algorithm_steps)}")
            
            return self.fig, path, success, final_stats
            
        except KeyboardInterrupt:
            print("\n⏹️ Görselleştirme kullanıcı tarafından durduruldu")
            return None
        except Exception as e:
            print(f"\n💥 Görselleştirme hatası: {e}")
            return None


class StepByStepVisualizer:
    """Adım adım interactive visualizer"""
    
    def __init__(self, graph, astar_algorithm):
        self.graph = graph
        self.astar = astar_algorithm
        self.realtime_viz = RealtimeAStarVisualizer(graph, astar_algorithm)
    
    def run_step_by_step(self):
        """Adım adım çalıştırma"""
        print(f"\n🎮 Adım Adım A* Görselleştirmesi")
        print(f"📝 Her adımda ENTER'a basarak ilerleyin")
        print(f"📊 Graf boyutu: {self.graph.width}x{self.graph.height}")
        print(f"🧮 Heuristic: {self.astar.heuristic_name}")
        
        return self.realtime_viz.run_realtime_visualization(
            step_delay=0.1, 
            interactive_mode=True
        )
    
    def run_auto_speed(self, speed="normal"):
        """Otomatik hızlı çalıştırma"""
        speed_settings = {
            "slow": 0.5,
            "normal": 0.2,
            "fast": 0.1,
            "very_fast": 0.05,
            "lightning": 0.01
        }
        
        delay = speed_settings.get(speed, 0.2)
        
        print(f"\n⚡ Otomatik A* Görselleştirmesi ({speed} hız)")
        print(f"⏱️ Adım arası gecikme: {delay} saniye")
        
        return self.realtime_viz.run_realtime_visualization(
            step_delay=delay, 
            interactive_mode=False
        )


def demo_realtime_astar():
    """Real-time A* demo fonksiyonu"""
    from graph import Graph
    from astar import AStar
    
    print("🎬 Real-time A* Demo")
    print("=" * 50)
    
    # Graf oluştur
    graph = Graph(30, 20)  # Daha küçük graf - görselleştirme için
    graph.create_random_walls(0.3)
    graph.set_start(2, 2)
    graph.set_goal(27, 17)
    
    # A* algoritması
    astar = AStar(graph, 'euclidean')
    
    # Real-time visualizer
    visualizer = StepByStepVisualizer(graph, astar)
    
    print(f"\nSeçenekler:")
    print("1. Otomatik (Normal hız)")
    print("2. Otomatik (Hızlı)")
    print("3. Adım adım (Manuel)")
    
    try:
        choice = input("\nSeçiminiz (1-3): ").strip()
        
        if choice == "1":
            result = visualizer.run_auto_speed("normal")
        elif choice == "2":
            result = visualizer.run_auto_speed("fast")
        elif choice == "3":
            result = visualizer.run_step_by_step()
        else:
            print("Varsayılan: Normal hız")
            result = visualizer.run_auto_speed("normal")
        
        if result:
            print("\n✅ Demo tamamlandı! Pencereyi kapatabilirsiniz.")
            input("ENTER'a basın...")
    
    except Exception as e:
        print(f"Demo hatası: {e}")


if __name__ == "__main__":
    demo_realtime_astar()