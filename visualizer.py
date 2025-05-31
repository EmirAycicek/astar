"""
A* Algoritması Görselleştirme
Matplotlib kullanarak animasyonlu görselleştirme
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.colors import ListedColormap
import time


class AStarVisualizer:
    """A* algoritması görselleştirici sınıfı"""
    
    def __init__(self, graph, astar_algorithm):
        self.graph = graph
        self.astar = astar_algorithm
        self.fig = None
        self.ax = None
        self.grid_display = None
        self.animation = None
        
        # Renkler
        self.colors = {
            'empty': (1.0, 1.0, 1.0, 1.0),      # Beyaz
            'wall': (0.2, 0.2, 0.2, 1.0),       # Koyu gri
            'start': (0.0, 1.0, 0.0, 1.0),      # Yeşil
            'goal': (1.0, 0.0, 0.0, 1.0),       # Kırmızı
            'path': (0.0, 0.0, 1.0, 1.0),       # Mavi
            'explored': (0.8, 0.8, 0.0, 0.6),   # Sarı (şeffaf)
            'frontier': (1.0, 0.5, 0.0, 0.6),   # Turuncu (şeffaf)
            'current': (1.0, 0.0, 1.0, 1.0)     # Magenta
        }
        
        # Renk haritası
        self.color_map = ListedColormap([
            self.colors['empty'],
            self.colors['wall'], 
            self.colors['start'],
            self.colors['goal'],
            self.colors['explored'],
            self.colors['frontier'],
            self.colors['current'],
            self.colors['path']
        ])
        
        self.color_values = {
            'empty': 0,
            'wall': 1,
            'start': 2,
            'goal': 3,
            'explored': 4,
            'frontier': 5,
            'current': 6,
            'path': 7
        }
    
    def create_grid_array(self):
        """Graf durumunu numpy array'e çevir"""
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
    
    def update_grid_with_algorithm_state(self, grid, step_info=None, path=None):
        """Grid'i algoritma durumuna göre güncelle"""
        # Önce temiz grid'i al
        updated_grid = self.create_grid_array()
        
        if step_info:
            # Keşfedilen düğümleri işaretle
            if 'closed_set' in step_info:
                for node in step_info['closed_set']:
                    if node.node_type == 'empty':
                        updated_grid[node.y, node.x] = self.color_values['explored']
            
            # Frontier (open set) düğümlerini işaretle
            if 'open_set' in step_info:
                for node in step_info['open_set']:
                    if node.node_type == 'empty':
                        updated_grid[node.y, node.x] = self.color_values['frontier']
            
            # Mevcut düğümü işaretle
            if 'current' in step_info:
                current = step_info['current']
                if current.node_type == 'empty':
                    updated_grid[current.y, current.x] = self.color_values['current']
        
        # Yolu çiz
        if path:
            for node in path:
                if node.node_type == 'empty':
                    updated_grid[node.y, node.x] = self.color_values['path']
        
        # Başlangıç ve hedefi her zaman göster
        if self.graph.start_node:
            start = self.graph.start_node
            updated_grid[start.y, start.x] = self.color_values['start']
        
        if self.graph.goal_node:
            goal = self.graph.goal_node
            updated_grid[goal.y, goal.x] = self.color_values['goal']
        
        return updated_grid
    def setup_plot(self):
        """Plot'u kurulum"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # İlk grid'i oluştur
        grid = self.create_grid_array()
        
        self.grid_display = self.ax.imshow(
            grid, 
            cmap=self.color_map, 
            interpolation='nearest',
            vmin=0, 
            vmax=len(self.color_values)-1
        )
        
        # Eksenleri ayarla
        self.ax.set_xlim(-0.5, self.graph.width - 0.5)
        self.ax.set_ylim(-0.5, self.graph.height - 0.5)
        self.ax.set_aspect('equal')
        
        # Grid çizgileri
        self.ax.set_xticks(np.arange(-0.5, self.graph.width, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.graph.height, 1), minor=True)
        self.ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Başlık ve etiketler
        self.ax.set_title(f'A* Pathfinding - Heuristic: {self.astar.heuristic_name}', 
                         fontsize=14, fontweight='bold')
        self.ax.set_xlabel('X Koordinatı')
        self.ax.set_ylabel('Y Koordinatı')
        
        # Renk legendı
        self.create_legend()
        
        return self.fig, self.ax
    
    def create_legend(self):
        """Renk legendı oluştur"""
        from matplotlib.patches import Patch
        
        legend_elements = [
            Patch(facecolor=self.colors['empty'], label='Boş Alan'),
            Patch(facecolor=self.colors['wall'], label='Duvar'),
            Patch(facecolor=self.colors['start'], label='Başlangıç'),
            Patch(facecolor=self.colors['goal'], label='Hedef'),
            Patch(facecolor=self.colors['explored'], label='Keşfedilen'),
            Patch(facecolor=self.colors['frontier'], label='Sınır (Open Set)'),
            Patch(facecolor=self.colors['current'], label='Mevcut Düğüm'),
            Patch(facecolor=self.colors['path'], label='Bulunan Yol')
        ]
        
        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    def animate_algorithm(self, interval=200, save_gif=False, filename='astar_animation.gif'):
        """A* algoritmasını animasyonlu olarak çalıştır"""
        # Önce algoritma adımlarını hesapla
        path, success, stats = self.astar.find_path(step_by_step=True)
        
        if not self.astar.algorithm_steps:
            print("Algoritma adımları bulunamadı!")
            return None
        
        # Plot'u kur
        self.setup_plot()
        
        # İstatistik metni için text box
        stats_text = self.ax.text(0.02, 0.98, '', transform=self.ax.transAxes, 
                                 fontsize=10, verticalalignment='top',
                                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        def animate(frame):
            if frame >= len(self.astar.algorithm_steps):
                # Son frame - yolu göster
                if success:
                    grid = self.update_grid_with_algorithm_state(None, None, path)
                else:
                    grid = self.update_grid_with_algorithm_state(None, 
                                                               self.astar.algorithm_steps[-1], None)
                
                stats_text.set_text(f'Tamamlandı!\n'
                                   f'Yol Bulundu: {"Evet" if success else "Hayır"}\n'
                                   f'Keşfedilen Düğüm: {stats["nodes_explored"]}\n'
                                   f'Yol Uzunluğu: {stats["path_length"]}\n'
                                   f'Toplam Adım: {stats["total_steps"]}')
            else:
                # Normal frame
                step_info = self.astar.algorithm_steps[frame]
                grid = self.update_grid_with_algorithm_state(None, step_info, None)
                
                # İstatistikleri güncelle
                stats_text.set_text(f'Adım: {step_info["step"]}\n'
                                   f'Mevcut: ({step_info["current"].x}, {step_info["current"].y})\n'
                                   f'Open Set: {len(step_info.get("open_set", []))}\n'
                                   f'Closed Set: {len(step_info.get("closed_set", []))}\n'
                                   f'İşlem: {step_info.get("action", "Bilinmiyor")}')
            
            self.grid_display.set_array(grid)
            return [self.grid_display, stats_text]
        
        # Animasyonu oluştur
        total_frames = len(self.astar.algorithm_steps) + 10  # Son durumu göstermek için ekstra frame
        self.animation = animation.FuncAnimation(
            self.fig, animate, frames=total_frames, interval=interval, 
            blit=False, repeat=True
        )
        
        # GIF olarak kaydet
        if save_gif:
            print(f"Animasyon {filename} olarak kaydediliyor...")
            self.animation.save(filename, writer='pillow', fps=1000//interval)
            print("Animasyon kaydedildi!")
        
        return self.animation
    
    def show_final_result(self, path=None, stats=None):
        """Son sonucu statik olarak göster"""
        self.setup_plot()
        
        if path:
            grid = self.update_grid_with_algorithm_state(None, None, path)
        else:
            grid = self.create_grid_array()
        
        self.grid_display.set_array(grid)
        
        # İstatistikleri göster
        if stats:
            stats_text = f"""
                Algoritma İstatistikleri:
                - Yol Bulundu: {"Evet" if stats["path_found"] else "Hayır"}
                - Keşfedilen Düğüm: {stats["nodes_explored"]}
                - Yol Uzunluğu: {stats["path_length"]}
                - Kullanılan Heuristic: {stats["heuristic_used"]}
                - Toplam Adım: {stats["total_steps"]}
            """
            
            self.ax.text(0.02, 0.02, stats_text, transform=self.ax.transAxes, 
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        return self.fig
    
    def compare_heuristics_visualization(self, heuristic_list):
        """Farklı heuristic'leri karşılaştır"""
        results = {}
        
        for heuristic_name in heuristic_list:
            self.astar.change_heuristic(heuristic_name)
            path, success, stats = self.astar.find_path(step_by_step=False)
            results[heuristic_name] = {
                'path': path,
                'success': success,
                'stats': stats
            }
        
        # Karşılaştırma grafiği
        fig, axes = plt.subplots(2, len(heuristic_list)//2 + len(heuristic_list)%2, 
                                figsize=(15, 10))
        axes = axes.flatten() if len(heuristic_list) > 1 else [axes]
        
        for i, (heuristic_name, result) in enumerate(results.items()):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            if result['success']:
                grid = self.update_grid_with_algorithm_state(None, None, result['path'])
            else:
                grid = self.create_grid_array()
            
            im = ax.imshow(grid, cmap=self.color_map, interpolation='nearest',
                          vmin=0, vmax=len(self.color_values)-1)
            
            ax.set_title(f'{heuristic_name}\nYol: {result["stats"]["path_length"]}, '
                        f'Keşfedilen: {result["stats"]["nodes_explored"]}')
            ax.set_xticks([])
            ax.set_yticks([])
        
        # Boş eksenleri gizle
        for i in range(len(heuristic_list), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        return fig, results
    
    def save_image(self, filename, dpi=300):
        """Mevcut görüntüyü kaydet"""
        if self.fig:
            self.fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Görüntü {filename} olarak kaydedildi!")
    
    def show(self):
        """Görüntüyü göster"""
        plt.show()


class StatisticsVisualizer:
    """İstatistik görselleştirici"""
    
    @staticmethod
    def plot_performance_comparison(results_dict):
        """Performans karşılaştırma grafiği"""
        heuristics = list(results_dict.keys())
        nodes_explored = [results_dict[h]['stats']['nodes_explored'] for h in heuristics]
        path_lengths = [results_dict[h]['stats']['path_length'] for h in heuristics]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Keşfedilen düğüm sayısı
        bars1 = ax1.bar(heuristics, nodes_explored, color='skyblue', alpha=0.7)
        ax1.set_title('Keşfedilen Düğüm Sayısı')
        ax1.set_ylabel('Düğüm Sayısı')
        ax1.tick_params(axis='x', rotation=45)
        
        # Değerleri bars üzerine yaz
        for bar, value in zip(bars1, nodes_explored):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    str(value), ha='center', va='bottom')
        
        # Yol uzunluğu
        bars2 = ax2.bar(heuristics, path_lengths, color='lightcoral', alpha=0.7)
        ax2.set_title('Bulunan Yol Uzunluğu')
        ax2.set_ylabel('Yol Uzunluğu')
        ax2.tick_params(axis='x', rotation=45)
        
        # Değerleri bars üzerine yaz
        for bar, value in zip(bars2, path_lengths):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                    str(value), ha='center', va='bottom')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def create_performance_table(results_dict):
        """Performans tablosu oluştur"""
        print("\n" + "="*80)
        print("HEURİSTİC PERFORMANS KARŞILAŞTIRMASI")
        print("="*80)
        print(f"{'Heuristic':<20} {'Başarılı':<10} {'Keşfedilen':<12} {'Yol Uzunluğu':<15} {'Adım':<8}")
        print("-"*80)
        
        for heuristic, result in results_dict.items():
            success = "Evet" if result['success'] else "Hayır"
            explored = result['stats']['nodes_explored']
            path_len = result['stats']['path_length'] if result['success'] else 0
            steps = result['stats']['total_steps']
            
            print(f"{heuristic:<20} {success:<10} {explored:<12} {path_len:<15} {steps:<8}")
        
        print("="*80)